# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek, Andreas Bengtsson 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from pathlib import Path
import time
from uuid import uuid4
from starlette.config import Config
import Labber
import requests

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
LOGFILE_DOWNLOAD_POOL_DIRNAME = config(
    "LOGFILE_DOWNLOAD_POOL_DIRNAME", default="logfile_download_pool"
)
MSS_MACHINE_ROOT_URL = config(
    "MSS_MACHINE_ROOT_URL", default="http://qdp-git.mc2.chalmers.se:5000"
)
BCC_MACHINE_ROOT_URL = config(
    "BCC_MACHINE_ROOT_URL", default="http://qtl-bcc-1.qdp.chalmers.se:5000"
)

REST_API_MAP = {
    "result": "/result",
    "status": "/status",
    "timelog": "/timelog",
    "jobs": "/jobs",
    "logfiles": "/logfiles",
    "download_url": "/download_url",
}


def logfile_postprocess(logfile: Path):

    print(f"Postprocessing logfile {str(logfile)}")

    # move logfile to download area
    new_file_name = str(uuid4())
    new_file_name_with_suffix = new_file_name + ".hdf5"
    storage_location = Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME

    new_file_path = storage_location / LOGFILE_DOWNLOAD_POOL_DIRNAME
    new_file_path.mkdir(exist_ok=True)
    new_file = new_file_path / new_file_name_with_suffix

    logfile.replace(new_file)

    print(f"Created new file {str(new_file)}")

    ###############################################################
    # The below works/makes sense only for Qiskit/QObj jobs.
    # Currently no check for this is in place !!!!
    #   - one WA possibility is to use tags exactly as with job_id
    #   - proper solution is a job supervisor
    ###############################################################

    # extract System state
    memory = extract_system_state_as_hex(new_file)
    print(memory)

    # extract job_id
    job_id = extract_job_id(new_file)
    MSS_JOB = MSS_MACHINE_ROOT_URL + REST_API_MAP["jobs"] + "/" + job_id

    # NOTE: When MSS adds support for the 'whole job' update
    # this will just one PUT request
    # Memory could contain more than one experiment, for now just use index 0
    response = requests.put(MSS_JOB + REST_API_MAP["result"], json=memory[0])
    if response:
        print("Pushed result to MSS")

    response = requests.post(MSS_JOB + REST_API_MAP["timelog"], json="RESULT")
    if response:
        print("Updated job timelog on MSS")

    response = requests.put(MSS_JOB + REST_API_MAP["status"], json="DONE")
    if response:
        print("Updated job status on MSS to DONE")

    download_url = BCC_MACHINE_ROOT_URL + REST_API_MAP["logfiles"] + "/" + new_file_name
    print(f"Download url: {download_url}")
    response = requests.put(MSS_JOB + REST_API_MAP["download_url"], json=download_url)
    if response:
        print("Updated job download_url on MSS")


def extract_system_state_as_hex(logfile: Path):
    f = Labber.LogFile(logfile)
    raw_data = f.getData("State Discriminator - System state")
    memory = []
    for entry in raw_data:
        memory.append([hex(int(x)) for x in entry])
    return memory


def extract_job_id(logfile: Path):
    f = Labber.LogFile(logfile)
    job_id = f.getTags()[0]
    print(job_id)
    return job_id
