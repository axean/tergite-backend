# This code is part of Tergite
#
# (C) Copyright Nicklas Botö, Fabian Forslund 2022
# (C) Copyright David Wahlstedt 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from pathlib import Path

from settings import (
    DEFAULT_PREFIX,
    JOB_EXECUTION_POOL_DIRNAME,
    REDIS_CONNECTION,
    STORAGE_PREFIX_DIRNAME,
    STORAGE_ROOT,
)

from ....libs.store import Collection
from ....utils.queues import QueuePool
from ..dtos import Job
from ..service import Stage
from ..utils import get_rq_job_id, log_job_msg, move_file, update_job_stage
from .execution import job_execute

_EXEC_POOL_DIR = STORAGE_ROOT / STORAGE_PREFIX_DIRNAME / JOB_EXECUTION_POOL_DIRNAME


# preprocessing queue
rq_queues = QueuePool(prefix=DEFAULT_PREFIX, connection=REDIS_CONNECTION)


def job_register(job_file: Path) -> None:
    """Registers job in job supervisor"""
    print(f"Registering job file {str(job_file)}")

    job_id = job_file.stem
    jobs_db = Collection[Job](REDIS_CONNECTION, schema=Job)

    # update job's stage and timestamps at the beginning
    update_job_stage(jobs_db, job_id=job_id, stage=Stage.REG_W)
    log_job_msg(f"Registered entry for job {job_id}")

    # store the received file in the job upload pool
    new_file = move_file(job_file, new_folder=_EXEC_POOL_DIR)

    # add job to executing queue and notify job supervisor
    rq_job_id = get_rq_job_id(job_id, Stage.EXEC_Q)
    rq_queues.job_preprocessing_queue.enqueue(job_execute, new_file, job_id=rq_job_id)

    # update job's stage and timestamps at the end
    update_job_stage(jobs_db, job_id=job_id, stage=Stage.EXEC_Q)
