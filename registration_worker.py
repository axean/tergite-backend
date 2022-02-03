# This code is part of Tergite
#
# (C) Copyright Nicklas Botö, Fabian Forslund 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import shutil
import time
from pathlib import Path
from typing import TypeVar

from redis import Redis
from rq import Queue, Worker

import settings
from preprocessing_worker import job_preprocess
from job_supervisor import register, Location

# settings
DEFAULT_PREFIX = settings.DEFAULT_PREFIX
STORAGE_ROOT = settings.STORAGE_ROOT
STORAGE_PREFIX_DIRNAME = settings.STORAGE_PREFIX_DIRNAME
JOB_EXECUTION_POOL_DIRNAME = settings.JOB_EXECUTION_POOL_DIRNAME
JOB_REGISTRATION_POOL_DIRNAME = settings.JOB_REGISTRATION_POOL_DIRNAME

# redis connection
redis_connection = Redis()

# preprocessing queue
rq_job_preprocessing = Queue(
    DEFAULT_PREFIX + "_job_preprocessing", connection=redis_connection
)

message_server = None

def job_register(job_file: Path) -> None:
    """ Registers job in job supervisor """

    print(f"Registering job file {str(job_file)}")

    # communicate with job supervisor here
    # register(job_file)
    # message_server.send("register", _format_job(job_file))

    # add job to pre-processing queue and notify job supervisor 
    rq_job_preprocessing.enqueue(job_preprocess, job_file)
    # message_server.send("here", Location.PRE_PROC_Q)

    # store the recieved file in the job upload pool
    file_name = job_file
    file_path = Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME / JOB_REGISTRATION_POOL_DIRNAME
    file_path.mkdir(parents=True, exist_ok=True)
    new_file = file_path / file_name

    # save it
    job_file.file.seek(0)
    with new_file.open() as destination:
        shutil.copyfileobj(job_file, destination)
    job_file.file.close()


def _format_job(job: Path) -> str:
    """"Format job file for storage"""
    entry = {
        "name": "test",
        "priorities": {
            "global": 0,
            "local": {
                "pre_processing": 0,
                "execution": 0,
                "post_processing": 0
            }
        },
        "status": {
            "location": 1,
            "started": time.now(),
            "finished": None,
            "cancelled": {
                "time": None,
                "reason": None
            }
        },
        "job": job,
        "scenario": None,
        "logfiles": None,
        "results": None
    }

