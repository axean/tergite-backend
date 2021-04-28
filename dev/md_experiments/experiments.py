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


from fastapi import FastAPI
import motor.motor_asyncio
from redis import Redis
from rq import Queue
import Labber
from labberfrontend import LabberFrontend
from labberworker import labber_job
from pydantic import BaseModel
from datetime import timedelta
from random import randint
import settings

# settings
DEFAULT_PREFIX = settings.DEFAULT_PREFIX
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
    title="Backend Control Computer experiments",
    description="Controls Pingu qubits via REST API",
    version="0.0.1",
)


# routing
@app.get("/")
async def root():
    return {"message": "Welcome to BCC machine experiments"}


@app.get("/db_access")
async def test_db_access():
    # test DB access
    cursor = await collection.find_one({"value": {"$gt": 60}})
    #    cursor.pop('_id')
    #    pprint.pprint(cursor)

    # output
    return {"message": "Hello from " + DEFAULT_PREFIX + ", T1: " + str(cursor["value"])}


@app.post("/rec")
async def record(item: Item):
    return {"message": "Received id: " + str(item.id)}


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


# check we can read the logfile
#   f = Labber.LogFile(logfile)
#   log_channels = f.getLogChannels()
#   print("Log channels:")
#   for channel in log_channels:
#      print(channel["name"])
