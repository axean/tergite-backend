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
import redis

STORAGE_ROOT = settings.STORAGE_ROOT
LABBER_MACHINE_ROOT_URL = settings.LABBER_MACHINE_ROOT_URL
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL
CALIBRATION_SUPERVISOR_PORT = settings.CALIBRATION_SUPERVISOR_PORT
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

# data Location
#   = REG
#   | PRE_PROC
#   | EXEC
#   | PST_PROV
#   deriving Enum


def register(job: Path) -> None:
    """Add job entry to redis"""
    pass

# TODO: Add job supervisor query 
async def query_redis(query: str) -> str:
    result = await red.hget(query)
    pass
    

# NOTE: This message handling is WIP! The calibration loop does *not* depend on it.
# Current status: The messages are ariving, but very late.
async def handle_message(reader, writer):
    data = await reader.read(100)
    message = data.decode()
    addr = writer.get_extra_info("peername")

    print(f"Received {message!r} from {addr!r}")
    writer.close()

async def message_server():
    server = await asyncio.start_server(
        handle_message, LOCALHOST, CALIBRATION_SUPERVISOR_PORT)
    async with server:
        await server.serve_forever()


async def main():
    server_task = asyncio.create_task(message_server())
    calib_task = asyncio.create_task(check_calib_status())
    await server_task
    await calib_task


@unique
class LogLevel(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2


def log(message: str, level: LogLevel = LogLevel.INFO) -> None:
    """Save message to job supervisor log file"""

    logstring = f"[{time.now()}] {level.name}: {message}"
 
    file_path = Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME 
    file_path.mkdir(parents=True, exist_ok=True)
    store_file = file_path / JOB_SUPERVISOR_LOG

    with store_file.open("a") as destination:
        destination.write(logstring)

