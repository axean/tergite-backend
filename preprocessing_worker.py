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


from redis import Redis
from rq import Queue
from pathlib import Path
from execution_worker import job_execute
from job_supervisor import inform_location, Location
import settings


# settings
DEFAULT_PREFIX = settings.DEFAULT_PREFIX
STORAGE_ROOT = settings.STORAGE_ROOT
STORAGE_PREFIX_DIRNAME = settings.STORAGE_PREFIX_DIRNAME
JOB_EXECUTION_POOL_DIRNAME = settings.JOB_EXECUTION_POOL_DIRNAME

# redis connection
redis_connection = Redis()

rq_job_execution = Queue(DEFAULT_PREFIX + "_job_execution", connection=redis_connection)


def job_preprocess(job_file: Path):
    job_id = job_file.stem

    # Inform supervisor about job being in pre-processing worker
    inform_location(job_id, Location.PRE_PROC_W)

    # mimick job pre-processing
    # time.sleep(2)

    new_file_name = job_file.stem
    storage_location = Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME

    new_file_path = storage_location / JOB_EXECUTION_POOL_DIRNAME
    new_file_path.mkdir(exist_ok=True)
    new_file = new_file_path / new_file_name

    job_file.replace(new_file)

    rq_job_execution.enqueue(job_execute, new_file, job_id=job_id)

    # Inform supervisor about job moved to execution queue
    inform_location(job_id, Location.EXEC_Q)

    print(f"Moved the job file to {str(new_file)}")
