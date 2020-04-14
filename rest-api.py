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
from starlette.config import Config
from starlette.datastructures import URL
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

# .env configuration
config = Config(".env")
NAME = config("NAME", default="NO-NAME")
DB_URL = config("DB_URL", default="NO-DB-URL")

# mongodb
mongodb = motor.motor_asyncio.AsyncIOMotorClient(DB_URL)
db = mongodb["milestone1"]
collection = db["t1_mon"]

# logfile
logfile = pathlib.Path("/tmp/logfile.hdf5")

# redis connection
redis_conn = Redis()
# redis queue
q_high = Queue("high", connection=redis_conn)
q_mid = Queue("mid", connection=redis_conn)
q_low = Queue("low", connection=redis_conn)

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
    # test DB access
    cursor = await collection.find_one({"value": {"$gt": 60}})
    #    cursor.pop('_id')
    #    pprint.pprint(cursor)

    # output
    return {"message": "Hello from " + NAME + ", T1: " + str(cursor["value"])}


@app.post("/logfiles")
def upload_logfile(upload_file: UploadFile = File(...)):
    with logfile.open("wb") as destination:
        shutil.copyfileobj(upload_file.file, destination)

    upload_file.file.close()

    f = Labber.LogFile(logfile)
    log_channels = f.getLogChannels()
    print("Log channels:")
    for channel in log_channels:
        print(channel["name"])

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


class Item(BaseModel):
    description: str = None
    id: int


@app.post("/rec")
async def record(item: Item):
    return {"message": "Received id: " + str(item.id)}


@app.get("/rq-info")
async def get_rq_info():
    workers = Worker.all(connection=redis_conn)
    print(str(workers))
    if workers == []:
        return {"message": "No worker registered"}

    msg = "{"
    for worker in workers:
        msg += "hostname: " + str(worker.hostname) + ","
        msg += "pid: " + str(worker.pid)
    msg += "}"

    return {"message": msg}
