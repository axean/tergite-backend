# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020, 2021
# (C) Copyright David Wahlstedt 2021
# (C) Andreas Bengtsson 2020
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
import Labber
import requests
import settings
import redis
from syncer import sync


# settings
STORAGE_ROOT = settings.STORAGE_ROOT
STORAGE_PREFIX_DIRNAME = settings.STORAGE_PREFIX_DIRNAME
LOGFILE_DOWNLOAD_POOL_DIRNAME = settings.LOGFILE_DOWNLOAD_POOL_DIRNAME
MSS_MACHINE_ROOT_URL = settings.MSS_MACHINE_ROOT_URL
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL
CALIBRATION_SUPERVISOR_PORT = settings.CALIBRATION_SUPERVISOR_PORT

LOCALHOST = "localhost"


REST_API_MAP = {
    "result": "/result",
    "status": "/status",
    "timelog": "/timelog",
    "jobs": "/jobs",
    "logfiles": "/logfiles",
    "download_url": "/download_url",
}

red = redis.Redis(decode_responses=True)


def logfile_postprocess(logfile: Path):

    print(f"Postprocessing logfile {str(logfile)}")

    # move the logfile to logfile download pool area
    # TODO: This file change should preferably happen _after_ the
    # post-processing.
    new_file_name = Path(logfile).stem
    new_file_name_with_suffix = new_file_name + ".hdf5"
    storage_location = Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME

    new_file_path = storage_location / LOGFILE_DOWNLOAD_POOL_DIRNAME
    new_file_path.mkdir(exist_ok=True)
    new_file = new_file_path / new_file_name_with_suffix

    logfile.replace(new_file)

    print(f"Moved the logfile to {str(new_file)}")

    # the post-processing itself
    tags = extract_tags(new_file)
    script_name = get_script_name(tags)
    is_calibration_sup_job = get_is_calibration_sup_job(tags)
    job_id = get_job_id(tags)
    print("Tags:", tags)

    ###############################################################
    # The code below uses a WA based on Labber job tags.
    #   - proper solution is a job supervisor
    ###############################################################

    if script_name == "demodulation_scenario":
        pass
    elif script_name in ["qiskit_qasm_runner", "qasm_dummy_job"]:

        # extract System state
        labber_logfile = Labber.LogFile(new_file)
        memory = extract_system_state_as_hex(labber_logfile)

        # helper printout with first 5 outcomes
        print("Measurement results:")
        for experiment_memory in memory:
            s = str(experiment_memory[:5])
            if experiment_memory[5:6]:
                s = s.replace("]", ", ...]")
            print(s)

        MSS_JOB = str(MSS_MACHINE_ROOT_URL) + REST_API_MAP["jobs"] + "/" + job_id

        # NOTE: When MSS adds support for the 'whole job' update
        # this will be just one PUT request
        # Memory could contain more than one experiment, for now just use index 0
        response = requests.put(MSS_JOB + REST_API_MAP["result"], json=memory)
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

    elif script_name == "qasm_dummy_job":  # DW FIXME: dead code:
        labber_logfile = Labber.LogFile(new_file)
        q_states = extract_system_state_as_hex(labber_logfile)
        print(f"qubit states: {len(q_states)} lists of length {len(q_states[0])}")
        if len(q_states[0]) <= 5:
            print(q_states)
        shots = extract_shots(labber_logfile)
        print(f"shots: {shots}")
        max_qubits = extract_max_qubits(labber_logfile)
        print(f"max qubits used in experiments: {max_qubits}")
        qobj_id = extract_qobj_id(labber_logfile)
        print(f"qobj ID: {qobj_id}")

    else:
        print(f"Unknown script name {script_name}")
        print("Postprocessing failed")

    if is_calibration_sup_job:
        # Job requested by calibration_supervisor
        sync(postprocess_calibration(Labber.LogFile(new_file), script_name))

    print(f"Postprocessing ended for script type: {script_name}")
    # we can return an object with more information if needed
    return (job_id, script_name, is_calibration_sup_job)


def extract_system_state_as_hex(logfile: Labber.LogFile):
    raw_data = logfile.getData("State Discriminator 2 States - System state")
    memory = []
    for entry in raw_data:
        memory.append([hex(int(x)) for x in entry])
    return memory


def extract_shots(logfile: Labber.LogFile):
    return int(logfile.getData("State Discriminator 2 States - Shots", 0)[0])


def extract_max_qubits(logfile: Labber.LogFile):
    return int(
        logfile.getData("State Discriminator 2 States - Max no. of qubits used", 0)[0]
    )


def extract_qobj_id(logfile: Labber.LogFile):
    return logfile.getChannelValue("State Discriminator 2 States - QObj ID")


def extract_tags(logfile: Path):
    return Labber.LogFile(logfile).getTags()


def get_job_id(tags):
    return tags[0]


def get_script_name(tags):
    return tags[1]


def get_is_calibration_sup_job(tags):
    # The third tag, if present and set to True, indicates this was
    # requested by the calibration supervisor
    return len(tags) >= 3 and tags[2]


def process_demodulation(logfile: Labber.LogFile):
    """
    Processes the logfile of a calibration supervisor scenario
    For now, we just extract the job_id, just to demonstrate the comcept of
    extracting something.
    """
    tags = logfile.getTags()
    job_id = get_job_id(tags)
    red.set("results:job_id", job_id)


def process_res_spect(logfile: Labber.LogFile):
    """
    Processes the logfile of a calibration supervisor scenario
    For now, we just extract the job_id, just to demonstrate the comcept of
    extracting something.
    """
    tags = logfile.getTags()
    job_id = get_job_id(tags)
    red.set("results:job_id", job_id)


PROCESSING_METHODS = {
    "demodulation_scenario": process_demodulation,
    "resonator_spectroscopy": process_res_spect,
    # etc...
}


async def postprocess_calibration(logfile: Labber.LogFile, measurement_type):
    # TODO
    # extract results from logfile
    # store results in Redis
    # Process the log's data appropriately
    try:
        PROCESSING_METHODS[measurement_type](logfile)
    except KeyError:
        pass


def postprocessing_success_callback(job, connection, result, *args, **kwargs):
    # From logfile_postprocess:
    (job_id, script_name, is_calibration_sup_job) = result
    print(f"Job with ID {job_id}, {script_name=} has finished")
    if is_calibration_sup_job:
        print(f"Results available in Redis. Notifying calibration supervisor.")
        sync(notify_job_done(job_id))


async def notify_job_done(job_id: str):
    reader, writer = await asyncio.open_connection(
        LOCALHOST, CALIBRATION_SUPERVISOR_PORT
    )
    message = ("job_done:" + job_id).encode()
    print(f"notify_job_done: {message=}")
    writer.write(message)
    writer.close()
