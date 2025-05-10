# This code is part of Tergite
#
# (C) Nicklas Botö, Fabian Forslund 2022
# (C) Chalmers Next Labs 2025
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
# - Martin Ahindura, 2023, 2025

from redis import Redis
from rq.command import send_stop_job_command
from rq.exceptions import InvalidJobOperation
from rq.job import Job as RqJob

from ...libs.store import Collection
from ...utils.datetime import utc_now_str
from .dtos import (
    Job,
    JobStatus,
    LogLevel,
    Stage,
)
from .exc import JobAlreadyCancelled
from .utils import get_rq_job_id, log_job_msg


def cancel_job(redis: Redis, job_id: str, reason: str) -> None:
    """Cancels a job by its id

    Args:
        redis: the connection to the redis database
        job_id: the unique identifier of the job
        reason: the reason for canceling the job

    Raises:
        ItemNotFoundError: key '{key}' not found
        JobAlreadyCancelled: job '{job_id}' is already cancelled
    """
    jobs_db = Collection[Job](redis, schema=Job)
    job: Job = jobs_db.get_one((job_id,))

    if job.status != JobStatus.PENDING:
        msg = f"Job {job_id} has finished, cancellation cancelled"
        log_job_msg(msg, level=LogLevel.WARNING)
        return

    rq_job_ids = [get_rq_job_id(job_id, stage=stage) for stage in Stage]
    rq_jobs = RqJob.fetch_many(rq_job_ids, connection=redis)

    for rq_job in rq_jobs:
        if rq_job is None:
            continue

        # Depending on whether job is in a worker or queue,
        # call appropriate cancel method
        if rq_job.worker_name:
            send_stop_job_command(redis, rq_job.id)
        else:
            try:
                rq_job.cancel()
            except InvalidJobOperation:
                raise JobAlreadyCancelled(f"job {job_id} already cancelled")

    jobs_db.update(
        (job_id,),
        {
            "status": JobStatus.CANCELLED,
            "cancellation_reason": reason,
            "updated_at": utc_now_str(),
        },
    )

    log_job_msg(f"Job {job_id} cancelled due to {reason}")
