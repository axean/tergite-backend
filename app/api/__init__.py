# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020, 2021
# (C) Copyright Abdullah-Al Amin 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Modified:
#
# - Martin Ahindura 2023


import json
import shutil
from contextlib import asynccontextmanager
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import Body, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse, Response
from redis.client import Redis
from rq import Worker
from typing_extensions import Annotated

import settings

from ..libs.quantify.connector.server import QuantifyMessage, QuantifyMessageType
from ..services.auth import service as auth_service
from ..services.jobs import service as jobs_service
from ..services.jobs.workers.registration import job_register
from ..services.properties import service as props_service
from ..utils.queues import QueuePool
from .dependencies import (
    get_bearer_token,
    get_quantify_connection,
    get_quantify_connector,
    get_redis_connection,
    get_valid_credentials_dep,
    get_whitelisted_ip,
)
from .exc import InvalidJobIdInUploadedFileError, IpNotAllowedError

# settings
DEFAULT_PREFIX = settings.DEFAULT_PREFIX
STORAGE_ROOT = settings.STORAGE_ROOT
STORAGE_PREFIX_DIRNAME = settings.STORAGE_PREFIX_DIRNAME
LOGFILE_UPLOAD_POOL_DIRNAME = settings.LOGFILE_UPLOAD_POOL_DIRNAME
LOGFILE_DOWNLOAD_POOL_DIRNAME = settings.LOGFILE_DOWNLOAD_POOL_DIRNAME
JOB_UPLOAD_POOL_DIRNAME = settings.JOB_UPLOAD_POOL_DIRNAME

# dependencies
RedisDep = Annotated[Redis, Depends(get_redis_connection)]
QuantifyConnectorDep = Annotated[Connection, Depends(get_quantify_connection)]


# redis queues
rq_queues = QueuePool(prefix=DEFAULT_PREFIX, connection=get_redis_connection())


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # on startup
    quantify_process, quantify_conn = get_quantify_connector()

    yield
    # on shutdown
    quantify_conn.send(QuantifyMessage(type=QuantifyMessageType.CLOSE, payload=None))
    quantify_process.join()


# application
app = FastAPI(
    title="Backend Control Computer",
    description="Interfaces Quantum processor via REST API",
    version="2024.02.0",
    lifespan=lifespan,
)


@app.exception_handler(InvalidJobIdInUploadedFileError)
async def invalid_job_id_in_file_exception_handler(
    request: Request, exp: InvalidJobIdInUploadedFileError
):
    """A custom exception handler to handle InvalidJobIdInUploadedFileError.

    This handler is only here to maintain the original way of responding
    when the job_id in the uploaded file was non-existent or was not a
    proper UUID
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "failed"},
    )


@app.middleware("http")
async def limit_access_to_ip_whitelist(request: Request, call_next):
    """Limits access to only the given IP addresses in the white list.

    This middleware adds the 'whitelisted_ip' property to the request.state
    if the IP of the request is in the CLIENT_IP_WHITELIST.
    Some endpoints will raise an IpNotAllowedError if the 'whitelisted_ip'
    property does not exist. Others will ignore it and work normally.

    The endpoints that raise an IpNotAllowedError are those that are
    essentially private.

    Args:
        request: the current FastAPI request object
        call_next: the callback that calls the next middleware or route handler
    """
    ip = f"{request.client.host}"

    if ip in settings.CLIENT_IP_WHITELIST:
        request.state.whitelisted_ip = ip

    try:
        return await call_next(request)
    except IpNotAllowedError:
        # return an empty response mimicking 404
        return Response(status_code=status.HTTP_404_NOT_FOUND)


# routing
@app.get("/", dependencies=[Depends(get_whitelisted_ip)])
async def root():
    return {"message": "Welcome to BCC machine"}


@app.post("/auth")
async def register_credentials(
    body: auth_service.Credentials, redis_connection: RedisDep
):
    """Registers the credentials passed to it"""
    try:
        auth_service.save_credentials(redis_connection, payload=body)
    except auth_service.JobAlreadyExists as exp:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"{exp}")
    return {"message": "ok"}


@app.post("/jobs")
async def upload_job(
    quantify_conn: QuantifyConnectorDep,
    upload_file: UploadFile = File(...),
    credentials: auth_service.Credentials = Depends(
        get_valid_credentials_dep(expected_status=auth_service.JobStatus.REGISTERED)
    ),
):
    # store the received file in the job upload pool
    file_name = job_id = credentials.job_id
    file_path = Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME / JOB_UPLOAD_POOL_DIRNAME
    file_path.mkdir(parents=True, exist_ok=True)
    store_file = file_path / file_name

    if jobs_service.does_job_exist(job_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"job_id {job_id} already exists",
        )

    # save it
    upload_file.file.seek(0)
    with store_file.open("wb") as destination:
        shutil.copyfileobj(upload_file.file, destination)
    upload_file.file.close()

    # enqueue for registration
    rq_queues.job_registration_queue.enqueue(
        job_register,
        store_file,
        job_id=credentials.job_id + f"_{jobs_service.Location.REG_Q.name}",
        quantify_conn=quantify_conn,
    )
    return {"message": file_name}


@app.get("/jobs", dependencies=[Depends(get_whitelisted_ip)])
async def fetch_all_jobs():
    return jobs_service.fetch_all_jobs()


@app.get("/jobs/{job_id}", dependencies=[Depends(get_valid_credentials_dep())])
async def fetch_job(job_id: str):
    job = jobs_service.fetch_job(job_id)
    return {"message": job or f"job {job_id} not found"}


@app.get("/jobs/{job_id}/status", dependencies=[Depends(get_valid_credentials_dep())])
async def fetch_job_status(job_id: str):
    job_status = jobs_service.fetch_job(job_id, "status", format=True)
    return {"message": job_status or f"job {job_id} not found"}


@app.get("/jobs/{job_id}/result", dependencies=[Depends(get_valid_credentials_dep())])
async def fetch_job_result(job_id: str):
    job = jobs_service.fetch_job(job_id)

    if not job:
        return {"message": f"job {job_id} not found"}
    elif job["status"]["finished"]:
        return {"message": job["result"]}
    else:
        return {"message": "job has not finished"}


@app.delete("/jobs/{job_id}", dependencies=[Depends(get_valid_credentials_dep())])
async def remove_job(job_id: str):
    jobs_service.remove_job(job_id)


@app.post("/jobs/{job_id}/cancel", dependencies=[Depends(get_valid_credentials_dep())])
async def cancel_job(job_id: str, reason: Optional[str] = Body(None, embed=False)):
    print(f"Cancelling job {job_id}")
    jobs_service.cancel_job(job_id, reason)


@app.get(
    "/logfiles/{logfile_id}",
    dependencies=[Depends(get_valid_credentials_dep(job_id_field="logfile_id"))],
)
async def download_logfile(logfile_id: UUID):
    file_name = f"{logfile_id}.hdf5"
    file = (
        Path(STORAGE_ROOT)
        / STORAGE_PREFIX_DIRNAME
        / LOGFILE_DOWNLOAD_POOL_DIRNAME
        / file_name
    )

    if file.exists():
        return FileResponse(file)
    else:
        return {"message": "logfile not found"}


# FIXME: this endpoint might be unnecessary going forward or might need to return proper JSON data
@app.get("/rq-info", dependencies=[Depends(get_whitelisted_ip)])
async def get_rq_info(redis_connection: RedisDep):
    workers = Worker.all(connection=redis_connection)
    print(str(workers))
    if len(workers) == 0:
        return {"message": "No worker registered"}

    msg = "{"
    for worker in workers:
        msg += "hostname: " + str(worker.hostname) + ","
        msg += "pid: " + str(worker.pid)
    msg += "}"

    return {"message": msg}


@app.get("/backend_properties", dependencies=[Depends(get_whitelisted_ip)])
async def create_current_snapshot():
    return props_service.create_backend_snapshot()


# FIXME: this endpoint might be unnecessary
@app.get("/web-gui", dependencies=[Depends(get_whitelisted_ip)])
async def get_snapshot(redis_connection: RedisDep):
    snapshot = redis_connection.get("current_snapshot")
    return json.loads(snapshot)


# FIXME: this endpoint might be unnecessary
@app.get("/web-gui/config", dependencies=[Depends(get_whitelisted_ip)])
async def web_config(redis_connection: RedisDep):
    snapshot = redis_connection.get("config")
    return json.loads(snapshot)
