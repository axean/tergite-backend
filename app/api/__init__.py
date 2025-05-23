# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020, 2021
# (C) Copyright Abdullah-Al Amin 2022
# (C) Copyright Chalmers Next Labs 2025
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


from uuid import UUID

from fastapi import Body, Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.requests import Request
from fastapi.responses import FileResponse, Response
from redis.client import Redis
from typing_extensions import Annotated

from settings import (
    CLIENT_IP_WHITELIST,
    DEFAULT_PREFIX,
    JOB_UPLOAD_POOL_DIRNAME,
    LOGFILE_DOWNLOAD_POOL_DIRNAME,
    STORAGE_PREFIX_DIRNAME,
    STORAGE_ROOT,
)

from ..libs import device_parameters as props_lib
from ..libs.device_parameters import get_backend_config, get_device_calibration_info
from ..services.auth import CredentialsAlreadyExists
from ..services.auth import service as auth_service
from ..services.jobs import service as jobs_service
from ..services.jobs.dtos import Job, JobStatus, Stage
from ..services.jobs.exc import JobAlreadyCancelled
from ..services.jobs.utils import get_rq_job_id
from ..services.jobs.workers.registration import job_register
from ..utils.api import save_uploaded_file, to_http_error
from ..utils.queues import QueuePool
from ..utils.store import Collection, ItemNotFoundError
from .dependencies import (
    get_bearer_token,
    get_redis_connection,
    get_valid_credentials_dep,
    get_whitelisted_ip,
    validate_job_file,
)
from .exc import InvalidJobIdInUploadedFileError, IpNotAllowedError

_JOB_UPLOAD_POOL = STORAGE_ROOT / STORAGE_PREFIX_DIRNAME / JOB_UPLOAD_POOL_DIRNAME
_LOG_FILE_POOL = STORAGE_ROOT / STORAGE_PREFIX_DIRNAME / LOGFILE_DOWNLOAD_POOL_DIRNAME

# dependencies
RedisDep = Annotated[Redis, Depends(get_redis_connection)]


# redis queues
rq_queues = QueuePool(prefix=DEFAULT_PREFIX, connection=get_redis_connection())


# application
app = FastAPI(
    title="Backend Control Computer",
    description="Interfaces Quantum processor via REST API",
    version="2025.03.2",
)

# exception handlers
app.add_exception_handler(InvalidJobIdInUploadedFileError, to_http_error(400))
app.add_exception_handler(CredentialsAlreadyExists, to_http_error(403))
app.add_exception_handler(ItemNotFoundError, to_http_error(404))
app.add_exception_handler(JobAlreadyCancelled, to_http_error(406))


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

    if ip in CLIENT_IP_WHITELIST:
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
    auth_service.save_credentials(redis_connection, payload=body)
    return {"message": "ok"}


@app.post("/jobs")
async def upload_job(
    redis_connection: RedisDep,
    upload_file: Annotated[UploadFile, Depends(validate_job_file)] = File(...),
    credentials: auth_service.Credentials = Depends(
        get_valid_credentials_dep(expected_status=JobStatus.PENDING)
    ),
):
    """Receives quantum jobs to process

    Args:
        redis_connection: the connection to the redis database
        upload_file: the quantum job file uploaded
        credentials: the (job_id, app_token) pair associated with this request
    """
    job_id = credentials.job_id
    jobs_db = Collection(redis_connection, schema=Job)
    if jobs_db.exists(job_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"job_id {job_id} already exists",
        )

    # save job file
    new_file_path = _JOB_UPLOAD_POOL / job_id
    job_file_path = save_uploaded_file(upload_file, target=new_file_path)

    # save job in database
    backend_config = get_backend_config()
    calibration_info = get_device_calibration_info(
        redis_connection, backend_config=backend_config
    )
    job = Job(
        job_id=job_id,
        device=backend_config.general_config.name,
        calibration_date=calibration_info.last_calibrated,
    )
    jobs_db.insert(job)

    # enqueue for registration
    rq_job_id = get_rq_job_id(job_id, Stage.REG_Q)
    rq_queues.job_registration_queue.enqueue(
        job_register, job_file_path, job_id=rq_job_id
    )
    return {"message": job_id}


@app.get("/jobs", dependencies=[Depends(get_whitelisted_ip)])
async def fetch_all_jobs(
    redis_connection: RedisDep,
):
    """Returns all available jobs

    Args:
        redis_connection: the connection to the redis database
    """
    jobs_db = Collection(redis_connection, schema=Job)
    data = jobs_db.get_all()
    # TODO: Paginate these in future
    return [item.model_dump(mode="json") for item in data]


@app.get("/jobs/{job_id}", dependencies=[Depends(get_valid_credentials_dep())])
async def fetch_job(redis_connection: RedisDep, job_id: str):
    """Returns a job of the given job_id"""
    jobs_db = Collection(redis_connection, schema=Job)
    job = jobs_db.get_one((job_id,))
    # TODO: Standardize the return schema here
    return {"message": job.model_dump(mode="json")}


@app.get("/jobs/{job_id}/status", dependencies=[Depends(get_valid_credentials_dep())])
async def fetch_job_status(redis_connection: RedisDep, job_id: str):
    """Returns the status of the given job of the given job_id

    Args:
        redis_connection: the connection to the redis database
        job_id: the unique identifier of the job
    """
    jobs_db = Collection(redis_connection, schema=Job)
    job: Job = jobs_db.get_one((job_id,))
    # TODO: Standardize the return schema here
    return {"message": job.status}


@app.get("/jobs/{job_id}/result", dependencies=[Depends(get_valid_credentials_dep())])
async def fetch_job_result(redis_connection: RedisDep, job_id: str):
    """Retrieves the result of the job if exists

    Args:
        redis_connection: the connection to the redis database
        job_id: the unique identifier of the job
    """
    jobs_db = Collection(redis_connection, schema=Job)
    job: Job = jobs_db.get_one((job_id,))
    if job.result is not None:
        # TODO: Standardize the return schema here
        return {"message": job.result.model_dump(mode="json")}

    # FIXME: this does not communicate well when the job has failed
    return {"message": "job has not finished"}


@app.delete("/jobs/{job_id}", dependencies=[Depends(get_valid_credentials_dep())])
async def remove_job(redis_connection: RedisDep, job_id: str):
    """Deletes the job of the given job_id

    Args:
        redis_connection: the connection to the redis database
        job_id: the unique identifier of the job
    """
    try:
        jobs_service.cancel_job(redis_connection, job_id=job_id, reason="deleting job")
    except JobAlreadyCancelled:
        pass

    jobs_db = Collection(redis_connection, schema=Job)
    jobs_db.delete_many([(job_id,)])
    return {"message": f"job {job_id} not found"}


@app.post("/jobs/{job_id}/cancel", dependencies=[Depends(get_valid_credentials_dep())])
async def cancel_job(
    redis_connection: RedisDep, job_id: str, reason: str = Body("", embed=False)
):
    """Cancels a given job's processing

    Args:
        redis_connection: the connection to the redis database
        job_id: the unique identifier of the job
        reason: reason for cancelling the job
    """
    print(f"Cancelling job {job_id}")
    jobs_service.cancel_job(redis_connection, job_id=job_id, reason=reason)


@app.get(
    "/logfiles/{logfile_id}",
    dependencies=[Depends(get_valid_credentials_dep(job_id_field="logfile_id"))],
)
async def download_logfile(logfile_id: UUID):
    """Downloads the job logfile

    Args:
        logfile_id: the id of the logfile usually the job id
    """
    file = (_LOG_FILE_POOL / str(logfile_id)).with_suffix(".hdf5")
    if file.exists():
        return FileResponse(file)
    return {"message": "logfile not found"}


@app.get("/static-properties", dependencies=[Depends(get_whitelisted_ip)])
async def get_static_properties(redis_connection: RedisDep):
    """Retrieves the device properties that are not changing"""
    return props_lib.get_device_info(redis_connection)


@app.get("/dynamic-properties", dependencies=[Depends(get_whitelisted_ip)])
async def get_dynamic_properties(redis_connection: RedisDep):
    """Retrieves the device properties that are changing with time i.e. calibration data"""
    return props_lib.get_device_calibration_info(redis_connection)
