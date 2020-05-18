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

from starlette.config import Config
from redis import Redis
from rq import Queue, Worker
import shutil
from pathlib import Path
from uuid import uuid4
import time
from execution_worker import job_execute


# config
config = Config(".env")
NAME = config("NAME", default="NO-NAME")
DB_URL = config("DB_URL", default="NO-DB-URL")
PREFIX = config("PREFIX", default="pingu")
API_PREFIX = config("API_PREFIX", default=PREFIX)
STORAGE_PREFIX_DIRNAME = config("STORAGE_PREFIX_DIRNAME", default=PREFIX)
STORAGE_ROOT = config("STORAGE_ROOT", default="/tmp")
JOB_UPLOAD_POOL_DIRNAME = config("JOB_UPLOAD_POOL_DIRNAME", default="job_upload_pool")
JOB_EXECUTION_POOL_DIRNAME = config(
    "JOB_EXECUTION_POOL_DIRNAME", default="job_execution_pool"
)

# redis connection
redis_connection = Redis()

rq_job_execution = Queue(PREFIX + "_job_execution", connection=redis_connection)


def job_preprocess(file):

    print(f"Preprocessing file {str(file)}")

    # mimick job pre-processing
    time.sleep(2)

    new_file_name = str(uuid4())
    storage_location = Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME

    new_file_path = storage_location / JOB_EXECUTION_POOL_DIRNAME
    new_file_path.mkdir(exist_ok=True)
    new_file = new_file_path / new_file_name

    file.replace(new_file)

    rq_job_execution.enqueue(job_execute, new_file)

    print(f"Created new file {str(new_file)}")
