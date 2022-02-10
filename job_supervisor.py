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
import asyncio
import time
import shutil
from pathlib import Path
import settings
from typing import Dict
import redis

STORAGE_ROOT = settings.STORAGE_ROOT
LABBER_MACHINE_ROOT_URL = settings.LABBER_MACHINE_ROOT_URL
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL
JOB_SUPERVISOR_LOG = settings.JOB_SUPERVISOR_LOG
STORAGE_PREFIX_DIRNAME = settings.STORAGE_PREFIX_DIRNAME

LOCALHOST = "localhost"

# Redis connection
# Maintains pools of
#   incoming jobs
#   registered jobs
#   pre-processed scenarios
#   incoming logfiles
#   logfiles for download
#   results

red = redis.Redis()

# TODO: Add job supervisor query 
async def query_redis(query: str) -> str:
    result = await red.hget("job_supervisor", query)
    return result


async def fetch_redis_entry(job_id: str) -> Dict[str, str]:
    """Query redis for job supervisor entry"""
    key = f"JS_{job_id}"
    return red.hget(key)

@unique
class Location(Enum):
    REG_Q      = 0
    REG_W      = 1
    PRE_PROC_Q = 2
    PRE_PROC_W = 3
    EXEC_Q     = 4
    EXEC_W     = 5
    PST_PROC_Q = 6
    PST_PROC_W = 7
    FINAL_Q    = 8
    FINAL_W    = 9

def register_job(job_id: str) -> None:
    """"Format job file for storage"""
    entry_id = f"JS_{job_id}"

    entry = {
        "id": entry_id,
        "priorities": {
            "global": 0,
            "local": {
                "pre_processing": 0,
                "execution": 0,
                "post_processing": 0
            }
        },
        "status": {
            "location": Location.REG_W,
            "started": time.now(),
            "finished": None,
            "cancelled": {
                "time": None,
                "reason": None
            }
        },
        "results": None
    }
    red.hset("job_supervisor", entry_id, entry)

    # log entry
    log(f"Registered entry for job {job_id}")


def update_job_entry(job_id: str, value, *keys) -> None:
    entry = query_redis(job_id)
    deep_update(entry, value, keys)
    red.hset("job_supervisor", job_id, entry)


def deep_update(dict, value, keys):
    for i, key in enumerate(keys):
        if key == keys[-1]:
            dict[key] = value
        else:
            sub = deep_update(dict[key], keys[i:], value)
            dict[key] = sub
    return dict


def cancel_job(job_id: str, reason: str) -> None:
    pass

def inform_results(job_id: str, results) -> None:
    """Upload results to redis"""

    update_job_entry(job_id, time.now(), "status", "finished")
    update_job_entry(job_id, results, "results")

    log("Job {job_id} finished with results")


async def inform_location(job_id: str, location: Location) -> None:
    """"Update job location"""
    
    parse_log: Dict[Location, str] = {
        Location.REG_Q      : "registration queue",
        Location.REG_W      : "registration worker",
        Location.PRE_PROC_Q : "pre-processing queue",
        Location.PRE_PROC_W : "pre-processing worker",
        Location.EXEC_Q     : "execution queue",
        Location.EXEC_W     : "execution worker",
        Location.PST_PROC_Q : "post-processing queue",
        Location.PST_PROC_W : "post-processing worker",
        Location.FINAL_Q    : "finalization queue",
        Location.FINAL_W    : "finalization worker"    
    }

    # update job position in redis
    job_entry = red.hget(job_id)
    job_entry["status"]["location"] = location
    red.hset(job_id, job_entry)

    # log updated job position
    log(f"{job_id} arrived at {parse_log[location]}")


async def main():
    server_task = asyncio.create_task()
    await server_task


@unique
class LogLevel(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2


def log(message: str, level: LogLevel = LogLevel.INFO) -> None:
    """Save message to job supervisor log file"""

    color = (
        '\033[0m', # color end
        '\033[0;33m', # yellow
        '\033[0;31m' # red
    )

    logstring = f"{color[level.value]}[{time.now()}] {level.name}: {message}{color[0]}"
 
    file_path = Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME 
    file_path.mkdir(parents=True, exist_ok=True)
    store_file = file_path / JOB_SUPERVISOR_LOG

    with store_file.open("a") as destination:
        destination.write(logstring)

