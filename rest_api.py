# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020, 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from redis import Redis
from rq import Queue, Worker
import shutil
from pathlib import Path
from uuid import uuid4, UUID
import json
from registration_worker import job_register
from postprocessing_worker import logfile_postprocess, postprocessing_success_callback
from job_supervisor import inform_location, Location
import settings
from utils.uuid import validate_uuid4_str

# settings
DEFAULT_PREFIX = settings.DEFAULT_PREFIX
STORAGE_ROOT = settings.STORAGE_ROOT
STORAGE_PREFIX_DIRNAME = settings.STORAGE_PREFIX_DIRNAME
LOGFILE_UPLOAD_POOL_DIRNAME = settings.LOGFILE_UPLOAD_POOL_DIRNAME
LOGFILE_DOWNLOAD_POOL_DIRNAME = settings.LOGFILE_DOWNLOAD_POOL_DIRNAME
JOB_UPLOAD_POOL_DIRNAME = settings.JOB_UPLOAD_POOL_DIRNAME


# redis connection
redis_connection = Redis()

# redis queues
rq_job_registration = Queue(
    DEFAULT_PREFIX + "_job_registration", connection=redis_connection
)

rq_logfile_postprocessing = Queue(
    DEFAULT_PREFIX + "_logfile_postprocessing", connection=redis_connection
)

# application
app = FastAPI(
    title="Backend Control Computer",
    description="Controls Pingu qubits via REST API",
    version="0.0.1",
)

# routing
@app.get("/")
async def root():
    return {"message": "Welcome to BCC machine"}


@app.post("/jobs")
async def upload_job(upload_file: UploadFile = File(...)):

    # get job_id and validate it
    job_dict = json.load(upload_file.file)
    job_id = job_dict.get("job_id", None)
    if job_id is None or validate_uuid4_str(job_id) is False:
        print("The job does not have a valid UUID4 job_id")
        return {"message": "failed"}

    # store the recieved file in the job upload pool
    file_name = job_id
    file_path = Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME / JOB_UPLOAD_POOL_DIRNAME
    file_path.mkdir(parents=True, exist_ok=True)
    store_file = file_path / file_name

    # save it
    upload_file.file.seek(0)
    with store_file.open("wb") as destination:
        shutil.copyfileobj(upload_file.file, destination)
    upload_file.file.close()

    # enqueue for registration
    rq_job_registration.enqueue(job_register, store_file)
    return {"message": file_name}


@app.get("/logfiles/{logfile_id}")
async def download_logfile(logfile_id: UUID):

    file_name = str(logfile_id) + ".hdf5"
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


@app.post("/logfiles")
def upload_logfile(upload_file: UploadFile = File(...)):

    print(f"Received logfile {upload_file.filename}")

    # store the recieved file in the logfile upload pool
    file_name = Path(upload_file.filename).stem
    file_path = (
        Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME / LOGFILE_UPLOAD_POOL_DIRNAME
    )
    file_path.mkdir(parents=True, exist_ok=True)
    store_file = file_path / file_name

    with store_file.open("wb") as destination:
        shutil.copyfileobj(upload_file.file, destination)

    upload_file.file.close()

    # enqueue for post-processing
    rq_logfile_postprocessing.enqueue(
        logfile_postprocess, store_file, on_success=postprocessing_success_callback
    )

    # inform supervisor
    inform_location(file_name, Location.PST_PROC_Q)

    return {"message": "ok"}


@app.get("/rq-info")
async def get_rq_info():

    workers = Worker.all(connection=redis_connection)
    print(str(workers))
    if workers == []:
        return {"message": "No worker registered"}

    msg = "{"
    for worker in workers:
        msg += "hostname: " + str(worker.hostname) + ","
        msg += "pid: " + str(worker.pid)
    msg += "}"

    return {"message": msg}
