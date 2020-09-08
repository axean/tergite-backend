# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020
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
import motor.motor_asyncio
import pprint
from redis import Redis
from rq import Queue, Worker
from labberfrontend import LabberFrontend
from labberworker import labber_job
from random import randint
from datetime import timedelta
from pydantic import BaseModel
import Labber
import shutil
import pathlib
from uuid import uuid4, UUID
from preprocessing_worker import job_preprocess
from postprocessing_worker import logfile_postprocess
import settings

# settings
DEFAULT_PREFIX = settings.DEFAULT_PREFIX
STORAGE_ROOT = settings.STORAGE_ROOT
STORAGE_PREFIX_DIRNAME = settings.STORAGE_PREFIX_DIRNAME
LOGFILE_UPLOAD_POOL_DIRNAME = settings.LOGFILE_UPLOAD_POOL_DIRNAME
LOGFILE_DOWNLOAD_POOL_DIRNAME = settings.LOGFILE_DOWNLOAD_POOL_DIRNAME
JOB_UPLOAD_POOL_DIRNAME = settings.JOB_UPLOAD_POOL_DIRNAME
DB_MACHINE_ROOT_URL = settings.DB_MACHINE_ROOT_URL

# mongodb
mongodb = motor.motor_asyncio.AsyncIOMotorClient(str(DB_MACHINE_ROOT_URL))
db = mongodb["milestone1"]
collection = db["t1_mon"]

# redis connection
redis_connection = Redis()
# redis queue
q_high = Queue("high", connection=redis_connection)
q_mid = Queue("mid", connection=redis_connection)
q_low = Queue("low", connection=redis_connection)

rq_job_preprocessing = Queue(
    DEFAULT_PREFIX + "_job_preprocessing", connection=redis_connection
)
rq_logfile_postprocessing = Queue(
    DEFAULT_PREFIX + "_logfile_postprocessing", connection=redis_connection
)


# pydantic models
class Item(BaseModel):
    description: str = None
    id: int


class Job(BaseModel):
    type: str
    name: str
    params: dict = None
    hdf5_log_extraction: dict = None
    hdf5_log_retention: str = None


# labber worker task
labber = LabberFrontend()

# application
app = FastAPI(
    title="Backend Control Computer",
    description="Controls Pingu qubits via REST API",
    version="0.0.1",
)

# routing
@app.get("/")
async def root():
    return {"message": "ok"}


@app.get("/db_access")
async def test_db_access():
    # test DB access
    cursor = await collection.find_one({"value": {"$gt": 60}})
    #    cursor.pop('_id')
    #    pprint.pprint(cursor)

    # output
    return {"message": "Hello from " + DEFAULT_PREFIX + ", T1: " + str(cursor["value"])}


@app.post("/jobs")
async def upload_job(upload_file: UploadFile = File(...)):

    # generate a unique file name
    uuid = uuid4()
    file_name = str(uuid)
    file_path = (
        pathlib.Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME / JOB_UPLOAD_POOL_DIRNAME
    )
    file_path.mkdir(parents=True, exist_ok=True)
    store_file = file_path / file_name

    # save it
    with store_file.open("wb") as destination:
        shutil.copyfileobj(upload_file.file, destination)
    upload_file.file.close()

    # enqueue for pre-processing
    rq_job_preprocessing.enqueue(job_preprocess, store_file)
    return {"message": file_name}


@app.get("/logfiles/{logfile_id}")
async def download_logfile(logfile_id: UUID):
    file_name = str(logfile_id) + ".hdf5"
    file = (
        pathlib.Path(STORAGE_ROOT)
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

    # generate a unique file name
    uuid = uuid4()
    file_name = str(uuid)
    file_path = (
        pathlib.Path(STORAGE_ROOT)
        / STORAGE_PREFIX_DIRNAME
        / LOGFILE_UPLOAD_POOL_DIRNAME
    )
    file_path.mkdir(parents=True, exist_ok=True)
    store_file = file_path / file_name

    with store_file.open("wb") as destination:
        shutil.copyfileobj(upload_file.file, destination)

    upload_file.file.close()

    # enqueue for post-processing
    rq_logfile_postprocessing.enqueue(logfile_postprocess, store_file)

    # check we can read the logfile
    # f = Labber.LogFile(logfile)
    # log_channels = f.getLogChannels()
    # print("Log channels:")
    # for channel in log_channels:
    #    print(channel["name"])

    return {"message": "ok"}


@app.get("/q")
async def root_q():

    # test redis queue
    rand = randint(0, 10)
    if rand < 6:
        job = q_high.enqueue(labber_job)
    elif rand < 8:
        job = q_mid.enqueue(labber.iJob, "1978", "0.53535")
    else:
        job = q_low.enqueue(labber.iJob, "1978", "0.53535")

    # enqueue_in() example
    # job = q_low.enqueue_in(timedelta(seconds=25), labber.addJob, args=( "1915", "0.1111"))

    # output
    return {"message": "Job id: " + job.id + " created"}


@app.post("/rec")
async def record(item: Item):
    return {"message": "Received id: " + str(item.id)}


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
