# This code is part of Tergite
#
# (C) Nicklas Botö, Fabian Forslund 2022
# (C) Martin Ahindura 2023
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
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, List, Tuple, Union

import requests
from pydantic import BaseModel

from app.services.jobs.dtos import (
    Job,
    JobResult,
    JobStatus,
    LogLevel,
    Stage,
    StageName,
    TimestampLabel,
    Timestamps,
)
from app.utils.datetime import utc_now_str
from app.utils.store import Collection
from settings import (
    BCC_MACHINE_ROOT_URL,
    JOB_SUPERVISOR_LOG,
    MSS_MACHINE_ROOT_URL,
    STORAGE_PREFIX_DIRNAME,
    STORAGE_ROOT,
)

_STAGE_TIMESTAMPS_MAP: Dict[Stage, Tuple[Tuple[StageName, TimestampLabel], ...]] = {
    Stage.REG_Q: (),
    Stage.REG_W: (("registration", "started"),),
    Stage.EXEC_Q: (("registration", "finished"),),
    Stage.EXEC_W: (("execution", "started"),),
    Stage.PST_PROC_Q: (("execution", "finished"),),
    Stage.PST_PROC_W: (("post_processing", "started"),),
    Stage.FINAL_Q: (("post_processing", "finished"), ("final", "started")),
    Stage.FINAL_W: (("final", "finished"),),
}

_STAGE_STATUS_MAP: Dict[Stage, JobStatus] = {
    Stage.REG_Q: JobStatus.PENDING,
    Stage.REG_W: JobStatus.PENDING,
    Stage.EXEC_Q: JobStatus.PENDING,
    Stage.EXEC_W: JobStatus.EXECUTING,
    Stage.PST_PROC_Q: JobStatus.EXECUTING,
    Stage.PST_PROC_W: JobStatus.EXECUTING,
    Stage.FINAL_Q: JobStatus.EXECUTING,
    Stage.FINAL_W: JobStatus.SUCCESSFUL,
}


def log_job_msg(message: str, level: LogLevel = LogLevel.INFO) -> None:
    """Save message to job supervisor log file.

    Args:
        message (str): message to log
        level (LogLevel, optional): log level of the message. Defaults to LogLevel.INFO.
    """
    # FIXME: Why a custom logger. Can't we use the normal logging
    color: Tuple[str, str, str] = (
        "\033[0m",  # color end
        "\033[0;33m",  # yellow
        "\033[0;31m",  # red
    )

    formatted_time = utc_now_str()

    logstring: str = (
        f"{color[level.value]}[{formatted_time}] {level.name}: {message}{color[0]}\n"
    )

    file_path = STORAGE_ROOT / STORAGE_PREFIX_DIRNAME
    file_path.mkdir(parents=True, exist_ok=True)
    store_file = file_path / JOB_SUPERVISOR_LOG

    with store_file.open("a") as destination:
        destination.write(logstring)


def move_file(file: Path, new_folder: Path, ext: str = "") -> Path:
    """Moves the file to a new folder

    Args:
        file: the file to move
        new_folder: the new folder to move to
        ext: the extension to attach to the final file

    Returns:
        the path to the new file
    """
    new_file_name = file.stem
    new_folder.mkdir(parents=True, exist_ok=True)
    new_file_path = (new_folder / new_file_name).with_suffix(ext)
    return file.replace(new_file_path)


def get_rq_job_id(quantum_job_id: str, stage: Stage) -> str:
    """Constructs an rq job id given the quantum job id and job stage

    Args:
        quantum_job_id: the job id of the quantum job
        stage: the stage at which we are at in the processing chain

    Returns:
        a string to be used as rq job id
    """
    return f"{quantum_job_id}_{stage.name}"


def update_job_stage(jobs_db: Collection[Job], job_id: str, stage: Stage) -> Job:
    """Updates the job's stage in the database

    This also updates the timestamps and the status of the job

    Args:
        jobs_db: the collection containing jobs
        job_id: the unique identifier of jobs
        stage: the stage to set on the job

    Returns:
        the updated job
    """
    key = (job_id,)
    job: Job = jobs_db.get_one(key)

    current_timestamp = utc_now_str()
    timestamps = _get_next_job_timestamps(
        job, next_stage=stage, current_time=current_timestamp
    )
    status = _get_next_status(job, next_stage=stage)

    return jobs_db.update(
        key,
        {
            "status": status,
            "stage": stage,
            "timestamps": timestamps,
            "updated_at": current_timestamp,
        },
    )


def log_job_failure(jobs_db: Collection[Job], job_id: str, reason: str) -> Job:
    """Logs the job in the db as failed

    Args:
        jobs_db: the collection containing job items
        job_id: the unique identifier of the job
        reason: the failure reason

    Returns:
        the updated job with its failure status
    """
    job = jobs_db.update(
        (job_id,),
        {
            "status": JobStatus.FAILED,
            "failure_reason": reason,
            "updated_at": utc_now_str(),
        },
    )

    log_job_msg(
        f"Job {job_id} failed at {job.stage.verbose_name} due to {reason}",
        level=LogLevel.ERROR,
    )

    return job


def update_job_results(
    jobs_db: Collection[Job], job_id: str, data: List[List[str]]
) -> Job:
    """Updates the results of the job and returns the updated job

    Args:
        jobs_db: the collection containing job items
        job_id: the unique identifier of the job
        data: the discriminated results from the quantum job

    Returns:
        the updated job
    """
    return jobs_db.update(
        (job_id,),
        {
            "status": JobStatus.SUCCESSFUL,
            "result": JobResult(memory=data),
            "download_url": f"{BCC_MACHINE_ROOT_URL}/logfiles/{job_id}",
            "updated_at": utc_now_str(),
        },
    )


def update_job_in_mss(
    mss_client: requests.Session, job_id: str, payload: Union[dict, Job]
) -> requests.Response:
    """Updates the job in MSS with the given payload

    Args:
        mss_client: the requests.Session that can query MSS
        job_id: the ID of the job
        payload: the new updates to apply to the given job in MSS

    Returns:
        the requests.Response received after request to MSS

    Raises:
        RuntimeError: Public API returned {resp.status_code}
    """
    data = payload
    if isinstance(payload, BaseModel):
        data = payload.model_dump(exclude_unset=True)

    url = f"{MSS_MACHINE_ROOT_URL}/jobs/{job_id}"
    resp = mss_client.put(url, json=data)

    if not resp.ok:
        try:
            message = resp.json()
        except JSONDecodeError:
            message = resp.text

        log_job_msg(
            f"failed to submit job to MSS\nstatus:{resp.status_code}\nresponse:{message}",
            level=LogLevel.ERROR,
        )
        raise RuntimeError(f"Public API returned {resp.status_code}")

    return resp


def _get_next_status(job: Job, next_stage: Stage) -> JobStatus:
    """Gets the next status given a job and the next stage

    Args:
        job: the quantum job
        next_stage: the next stage this job is to go to

    Returns:
        the next job status for that job
    """
    status = job.status
    if not status.is_terminal():
        status = _STAGE_STATUS_MAP[next_stage]
    return status


def _get_next_job_timestamps(
    job: Job, next_stage: Stage, current_time: str
) -> Timestamps:
    """Gets the next timestamps for the given job and the next stage

    Args:
        job: the quantum job
        next_stage: the next stage this job is to go to
        current_time: the current timestamp as a string

    Returns:
        the next timestamps for that job
    """
    timestamps = job.timestamps
    if timestamps is None:
        timestamps = Timestamps()

    new_timestamps = {
        stage_name: {timestamp_label: current_time}
        for stage_name, timestamp_label in _STAGE_TIMESTAMPS_MAP[next_stage]
    }

    return timestamps.with_updates(new_timestamps)
