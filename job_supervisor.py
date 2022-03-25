# This code is part of Tergite
#
# (C) Nicklas Botö, Fabian Forslund 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from enum import Enum, unique
from datetime import datetime
from pathlib import Path
import settings
from typing import List, Tuple, Dict, Any, Union
import redis
from rq.command import send_stop_job_command
import json

STORAGE_ROOT = settings.STORAGE_ROOT
LABBER_MACHINE_ROOT_URL = settings.LABBER_MACHINE_ROOT_URL
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL
JOB_SUPERVISOR_LOG = settings.JOB_SUPERVISOR_LOG
STORAGE_PREFIX_DIRNAME = settings.STORAGE_PREFIX_DIRNAME

LOCALHOST = "localhost"

# Redis connection
red = redis.Redis()

# Type hint contants
Entry = Dict[str, Any]
Result = Tuple[str, str]


@unique
class Location(Enum):
    """Job location in the BCC chain"""

    REG_Q = 0
    REG_W = 1
    PRE_PROC_Q = 2
    PRE_PROC_W = 3
    EXEC_Q = 4
    EXEC_W = 5
    PST_PROC_Q = 6
    PST_PROC_W = 7
    FINAL_Q = 8
    FINAL_W = 9


# Parse a location
STR_LOC: Dict[Location, str] = {
    Location.REG_Q: "registration queue",
    Location.REG_W: "registration worker",
    Location.PRE_PROC_Q: "pre-processing queue",
    Location.PRE_PROC_W: "pre-processing worker",
    Location.EXEC_Q: "execution queue",
    Location.EXEC_W: "execution worker",
    Location.PST_PROC_Q: "post-processing queue",
    Location.PST_PROC_W: "post-processing worker",
    Location.FINAL_Q: "finalization queue",
    Location.FINAL_W: "finalization worker",
}


class EnumEncoder(json.JSONEncoder):
    """Encodes children of enumerable classes"""

    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return json.JSONEncoder.default(self, obj)


class JobNotFound(Exception):
    """A job was not found on redis"""

    def __init__(self, job_id) -> None:
        self.job_id = job_id

    def __str__(self):
        return f"Job {self.job_id} not found"


def now() -> str:
    """Returns the current time formatted"""
    return datetime.now().isoformat()


def fetch_redis_entry(job_id: str) -> Entry:
    """Query redis for job supervisor entry."""
    entry = red.hget("job_supervisor", job_id)

    if not entry:
        log(f"Job {job_id} not found", level=LogLevel.ERROR)
        raise JobNotFound(job_id)

    return _load_json(entry)


def register_job(job_id: str) -> None:
    # job entry skeleton
    entry = {
        "id": job_id,
        "priorities": {
            "global": 0,
            "local": {"pre_processing": 0, "execution": 0, "post_processing": 0},
        },
        "status": {
            "location": Location.REG_W,
            "started": now(),
            "finished": None,
            "cancelled": {"time": None, "reason": None},
            "failed": {"time": None, "reason": None},
        },
        "result": None,
    }
    red.hset("job_supervisor", job_id, json.dumps(entry, cls=EnumEncoder))

    # log entry
    log(f"Registered entry for job {job_id}")


def update_job_entry(job_id: str, value: Any, *keys: List[str]) -> None:
    """Updates job dict entry with given key(s).

    Args:
        job_id (str): identifier of job to be updated
        value (Any): new value of dictionary entry
        keys (List[str]): nested keys of dictionary
    """
    entry: Entry = fetch_redis_entry(job_id)
    _deep_update(entry, value, keys)
    red.hset("job_supervisor", job_id, json.dumps(entry, cls=EnumEncoder))


def _deep_update(dict: Entry, value: Any, keys: List[str]) -> Entry:
    """Update a nested dictionary.

    Args:
        dict (Entry): nested dictionary to be updated
        value (Any): value to set
        keys (List[str]): nested keys

    Returns:
        Entry: the updated entry
            (note that this is the reference to the input dictionary)
    """
    if len(keys) == 1:
        dict[keys[0]] = value
        return dict

    return {keys[0]: _deep_update(dict[keys[0]], value, keys[1:])}


def cancel_job(job_id: str, reason: str) -> None:
    """Cancels a job by its id, regardless of which Queue it is in."""

    # This may not be the right function to use, but it works
    send_stop_job_command(red, job_id)

    update_job_entry(job_id, {"time": now(), "reason": reason}, "status", "cancelled")

    if reason:
        log_message = f"Job {job_id} cancelled due to {reason}"
    else:
        log_message = f"Job {job_id} cancelled."

    log(log_message)


def inform_result(job_id: str, result: Result) -> None:
    """Upload result to redis."""
    update_job_entry(job_id, now(), "status", "finished")
    update_job_entry(job_id, result, "result")

    log(f"Job {job_id} finished with result")


def inform_location(job_id: str, location: Location) -> None:
    """ "Update job location."""
    update_job_entry(job_id, location, "status", "location")

    # log updated job position
    log(f"{job_id} arrived at {STR_LOC[location]}")


def inform_failure(job_id: str, reason: str = None) -> None:
    """Inform job supervisor that a job has failed

    Args:
        job_id (str): Identifier of the job that failed.
        reason (str, optional): Reason for failure. Defaults to None.
    """
    entry: Entry = fetch_redis_entry(job_id)
    update_job_entry(job_id, {"time": now(), "reason": reason}, "status", "failed")

    location = entry["status"]["location"]

    if reason:
        log_message: str = (
            f"Job {job_id} failed at {STR_LOC[location]} due to {reason}."
        )
    else:
        log_message: str = f"Job {job_id} failed at {STR_LOC[location]}."

    log(log_message, level=LogLevel.ERROR)


def _load_json(json_str: str) -> Entry:
    """Loads json string into an entry.
    
    Args:
        json_str (str): The json string to load.

    Returns:
        Entry: The loaded entry.
    """
    entry: Entry = json.loads(json_str)
    location = entry["status"]["location"]
    entry["status"]["location"] = Location(location)
    return entry


def fetch_all_jobs() -> List[Entry]:
    """Fetches all jobs from redis

    Returns:
        List[Entry]: The list of job entires.
    """
    entries = red.hgetall("job_supervisor")
    return {k: _load_json(v) for k, v in entries.items()}


def fetch_job(
    job_id: str, key: str = None, format: bool = False
) -> Union[Entry, Result]:
    """Fetch specific job from redis

    Args:
        job_id (str): Identifier of job to fetch
        key (str, optional): Only fetch this key. Defaults to None.
        format (bool, optional): Formats location value. Defaults to False.
    """
    entry = fetch_redis_entry(job_id)

    if format and key == "status":
        entry[key]["location"] = STR_LOC[entry[key]["location"]]

    return entry[key] if key else entry


def remove_job(job_id: str) -> None:
    """Remove job entry from redis

    Args:
        job_id (str): Identifier of the job to be deleted
    """
    cancel_job(job_id, f"Job ID {job_id} was deleted")
    red.hdel("job_supervisor", job_id)
    log(f"Job {job_id} was deleted")


@unique
class LogLevel(Enum):
    """Log level of job supervisor log messages"""

    INFO = 0
    WARNING = 1
    ERROR = 2


def log(message: str, level: LogLevel = LogLevel.INFO) -> None:
    """Save message to job supervisor log file.

    Args:
        message (str): message to log
        level (LogLevel, optional): log level of the message. Defaults to LogLevel.INFO.
    """
    color: Tuple(str, str, str) = (
        "\033[0m",  # color end
        "\033[0;33m",  # yellow
        "\033[0;31m",  # red
    )

    current_time = datetime.now()
    formatted_time = datetime.strftime(current_time, "%Y-%m-%d %H:%M:%S:%f")

    logstring: str = (
        f"{color[level.value]}[{formatted_time}] {level.name}: {message}{color[0]}\n"
    )

    file_path = Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME
    file_path.mkdir(parents=True, exist_ok=True)
    store_file = file_path / JOB_SUPERVISOR_LOG

    with store_file.open("a") as destination:
        destination.write(logstring)
