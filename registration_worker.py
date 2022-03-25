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
from job_supervisor import register_job, inform_location, Location

# settings
DEFAULT_PREFIX = settings.DEFAULT_PREFIX
STORAGE_ROOT = settings.STORAGE_ROOT
STORAGE_PREFIX_DIRNAME = settings.STORAGE_PREFIX_DIRNAME
JOB_EXECUTION_POOL_DIRNAME = settings.JOB_EXECUTION_POOL_DIRNAME
JOB_PRE_PROC_POOL_DIRNAME = settings.JOB_PRE_PROC_POOL_DIRNAME

# redis connection
redis_connection = Redis()

# preprocessing queue
rq_job_preprocessing = Queue(
    DEFAULT_PREFIX + "_job_preprocessing", connection=redis_connection
)


def job_register(job_file: Path) -> None:
    """ Registers job in job supervisor """
    job_id = job_file.stem

    # inform job supervisor about job registration
    print(f"Registering job file {str(job_file)}")
    register_job(job_id)

    # store the received file in the job upload pool
    new_file_name = job_file.stem
    new_file_path = Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME / JOB_PRE_PROC_POOL_DIRNAME
    new_file_path.mkdir(exist_ok=True)
    new_file = new_file_path / new_file_name

    job_file.replace(new_file)
    # add job to pre-processing queue and notify job supervisor
    rq_job_preprocessing.enqueue(job_preprocess, new_file, job_id=job_id)

    inform_location(job_id, Location.PRE_PROC_Q)