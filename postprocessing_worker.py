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
import asyncio
import time
from uuid import uuid4
import Labber
import requests
import settings

# settings
STORAGE_ROOT = settings.STORAGE_ROOT
STORAGE_PREFIX_DIRNAME = settings.STORAGE_PREFIX_DIRNAME
LOGFILE_DOWNLOAD_POOL_DIRNAME = settings.LOGFILE_DOWNLOAD_POOL_DIRNAME
MSS_MACHINE_ROOT_URL = settings.MSS_MACHINE_ROOT_URL
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL

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
    tags = extract_tags(new_file)
    script_name = get_script_name(tags)
    job_id = get_job_id(tags)
    print("Tags:", tags)

    ###############################################################
    # The code below uses a WA based on Labber job tags.
    #   - proper solution is a job supervisor
    ###############################################################

    if script_name == "demodulation_scenario":
        pass

    elif script_name == "calibration":
        asyncio.run(postprocess_calibration())

    elif script_name == "qiskit_qasm_runner":

        # extract System state
        new_file = Labber.LogFile(new_file)
        memory = extract_system_state_as_hex(new_file)
        print(memory)

        MSS_JOB = str(MSS_MACHINE_ROOT_URL) + REST_API_MAP["jobs"] + "/" + job_id

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

        download_url = (
            str(BCC_MACHINE_ROOT_URL) + REST_API_MAP["logfiles"] + "/" + new_file_name
        )
        print(f"Download url: {download_url}")
        response = requests.put(
            MSS_JOB + REST_API_MAP["download_url"], json=download_url
        )
        if response:
            print("Updated job download_url on MSS")
    elif script_name == "qasm_dummy_job":
        new_file = Labber.LogFile(new_file)
        q_states = extract_system_state_as_hex(new_file)
        print(f"qubit states: {len(q_states)} lists of length {len(q_states[0])}")
        shots = extract_shots(new_file)
        print(f"shots: {shots}")
        max_qubits = extract_max_qubits(new_file)
        print(f"max qubits used in experiments: {max_qubits}")
        qobj_id = extract_qobj_id(new_file)
        print(f"qobj ID: {qobj_id}")
    else:
        print(f"Unknown script name {script_name}")
        print("Postprocessing failed")

    print(f"Postprocessing ended for script type: {script_name}")


def extract_system_state_as_hex(logfile: Labber.LogFile):
    raw_data = logfile.getData("State Discriminator - System state")
    memory = []
    for entry in raw_data:
        memory.append([hex(int(x)) for x in entry])
    return memory


def extract_shots(logfile: Labber.LogFile):
    return int(logfile.getData("State Discriminator - Shots", 0)[0])


def extract_max_qubits(logfile: Labber.LogFile):
    return int(logfile.getData("State Discriminator - Max no. of qubits used", 0)[0])


def extract_qobj_id(logfile: Labber.LogFile):
    return logfile.getChannelValue("State Discriminator - QObj ID")


def extract_tags(logfile: Path):
    return Labber.LogFile(logfile).getTags()


def get_job_id(tags):
    return tags[0]


def get_script_name(tags):
    return tags[1]


async def postprocess_calibration():
    # TODO
    # extract results from logfile
    # store results in Redis

    # inform calibration deamon that the results are available
    reader, writer = await asyncio.open_connection("127.0.0.1", 8888)

    message = "Calibration routine finished. Results available in Redis"
    print(f"Send: {message!r}")
    writer.write(message.encode())

    writer.close()
