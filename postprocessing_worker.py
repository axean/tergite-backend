# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020, 2021
# (C) Copyright David Wahlstedt 2021, 2022
# (C) Copyright Abdullah Al Amin 2021, 2022
# (C) Copyright Axel Andersson 2022
# (C) Andreas Bengtsson 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import argparse
import asyncio
import enums
from pathlib import Path
import settings
from typing import Any, List, Dict

import redis
import requests
from syncer import sync

import Labber

from analysis import (
    extract_resonance_freqs,
    fit_oscillation_idx,
    fit_resonator,
    fit_resonator_idx,
    gaussian_fit_idx,
)
from job_supervisor import (
    inform_failure,
    inform_location,
    inform_result,
    Location,
)
import tqcsf.file

# Storage settings

STORAGE_ROOT = settings.STORAGE_ROOT
STORAGE_PREFIX_DIRNAME = settings.STORAGE_PREFIX_DIRNAME
LOGFILE_DOWNLOAD_POOL_DIRNAME = settings.LOGFILE_DOWNLOAD_POOL_DIRNAME

# Connectivity settings

MSS_MACHINE_ROOT_URL = settings.MSS_MACHINE_ROOT_URL
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL
CALIBRATION_SUPERVISOR_PORT = settings.CALIBRATION_SUPERVISOR_PORT

LOCALHOST = "localhost"

# REST API

REST_API_MAP = {
    "result": "/result",
    "status": "/status",
    "timelog": "/timelog",
    "jobs": "/jobs",
    "logfiles": "/logfiles",
    "download_url": "/download_url",
}


# Redis connection

red = redis.Redis(decode_responses=True)

# =========================================================================
# Post-processing

def logfile_postprocess(logfile: Path, *, logfile_type: enums.LogfileType = enums.LogfileType.LABBER_LOGFILE):

    print(f"Postprocessing logfile {str(logfile)}")

    # Move the logfile to logfile download pool area
    # TODO: This file change should preferably happen _after_ the
    # post-processing.
    new_file_name = Path(logfile).stem  # This is the job_id
    new_file_name_with_suffix = new_file_name + ".hdf5"
    storage_location = Path(STORAGE_ROOT) / STORAGE_PREFIX_DIRNAME

    new_file_path = storage_location / LOGFILE_DOWNLOAD_POOL_DIRNAME
    new_file_path.mkdir(exist_ok=True)
    new_file = new_file_path / new_file_name_with_suffix

    logfile.replace(new_file)

    print(f"Moved the logfile to {str(new_file)}")

    # Inform job supervisor
    inform_location(new_file_name, Location.PST_PROC_W)

    # The post-processing itself
    if logfile_type == enums.LogfileType.TQC_STORAGE:
        print("Identified TQC storage file, reading file using tqcsf")
        sf = tqcsf.file.StorageFile(new_file, mode="r")
        return postprocess_tqcsf(sf)
    else:
        return postprocess(new_file)


# =========================================================================
# Post-processing helpers in PROCESSING_METHODS
# =========================================================================

# Dummy post-processing of signal demodulation
def process_demodulation(logfile: Path) -> Any:
    labber_logfile = Labber.LogFile(logfile)
    (job_id, _, _) = get_metainfo(labber_logfile)
    return job_id

# Qasm job example
def process_qiskit_qasm_runner_qasm_dummy_job(logfile: Path) -> Any:
    labber_logfile = Labber.LogFile(logfile)
    (job_id, _, _) = get_metainfo(labber_logfile)

    # Extract System state
    memory = extract_system_state_as_hex(labber_logfile)

    update_mss_and_bcc(memory, job_id)

    # DW: I guess something else should be returned? memory or parts of it?
    return job_id


def postprocess_tqcsf(sf: tqcsf.file.StorageFile) -> tuple:

    if sf.meas_level == tqcsf.file.MeasLvl.DISCRIMINATED:
        pass # TODO

    elif sf.meas_level == tqcsf.file.MeasLvl.INTEGRATED:
        pass # TODO

    elif sf.meas_level == tqcsf.file.MeasLvl.RAW:
        pass # TODO

    else:
        pass

    return (sf.job_id, "pulse_schedule", False)

# VNA resonator spectroscopy
def process_res_spect_vna_phase_1(logfile: Path) -> Any:
    return fit_resonator(logfile)

def process_res_spect_vna_phase_2(logfile: Path) -> Any:
    return fit_resonator_idx(logfile, [0,50])

# Pulsed resonator spectroscopy
def process_pulsed_res_spect(logfile: Path) -> Any:
    return fit_resonator_idx(logfile, [0])

# Two-tone
def process_two_tone(logfile: Path) -> Any:
    # fit qubit spectra
    return gaussian_fit_idx(logfile, [0])

# Rabi
def process_rabi(logfile: Path) -> Any:
    # fit rabi oscillation
    fits = fit_oscillation_idx(logfile, [0])
    return [res['period'] for res in fits]

# Ramsey
def process_ramsey(logfile: Path) -> Any:
    # fit ramsey oscillation
    fits = fit_oscillation_idx(logfile, [0])
    return [res['freq'] for res in fits]


# =========================================================================
# Post-processing entry point
# =========================================================================

PROCESSING_METHODS = {
    "resonator_spectroscopy": process_res_spect_vna_phase_1,
    "fit_resonator_spectroscopy": process_res_spect_vna_phase_2,
    "pulsed_resonator_spectroscopy": process_pulsed_res_spect,
    "pulsed_two_tone_qubit_spectroscopy": process_two_tone,
    "rabi_qubit_pi_pulse_estimation": process_rabi,
    "ramsey_qubit_freq_correction": process_ramsey,
    "demodulation_scenario": process_demodulation,
    "qiskit_qasm_runner": process_qiskit_qasm_runner_qasm_dummy_job,
    "qasm_dummy_job": process_qiskit_qasm_runner_qasm_dummy_job,
}

def postprocess(logfile: Path):

    labber_logfile = Labber.LogFile(logfile)
    (job_id, script_name, is_calibration_sup_job) = get_metainfo(labber_logfile)

    postproc_fn = PROCESSING_METHODS.get(script_name)

    print(
        f"Starting postprocessing for script: {script_name}, {job_id=}, {is_calibration_sup_job=}"
    )

    if postproc_fn:
        results = postproc_fn(logfile)
    else:
        print(f"Unknown script name {script_name}")
        print("Postprocessing failed")  # TODO: take care of this case
        results = None

        # Inform job supervisor about failure
        inform_failure(job_id, "Unknown script name")

    print(f"Postprocessing ended for script type: {script_name}, {job_id=}, {is_calibration_sup_job=}")
    red.set(f"postproc:results:{job_id}", str(results))
    return (job_id, script_name, is_calibration_sup_job)


# =========================================================================
# Post-processing success callback with helper
# =========================================================================

async def notify_job_done(job_id: str):
    reader, writer = await asyncio.open_connection(
        LOCALHOST, CALIBRATION_SUPERVISOR_PORT
    )
    message = ("job_done:" + job_id).encode()
    print(f"notify_job_done: {message=}")
    writer.write(message)
    writer.close()

def postprocessing_success_callback(job, connection, result, *args, **kwargs):
    # From logfile_postprocess:
    (job_id, script_name, is_calibration_sup_job) = result

    # Inform job supervisor about results
    inform_result(job_id, result)

    print(f"Job with ID {job_id}, {script_name=} has finished")
    if is_calibration_sup_job:
        print(f"Results available in Redis. Notifying calibration supervisor.")
        sync(notify_job_done(job_id))



# =========================================================================
# Extraction helpers
# =========================================================================

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


def extract_tags(logfile: Labber.LogFile):
    return logfile.getTags()


def get_job_id(tags):
    return tags[0]


def get_script_name(tags):
    return tags[1]


def get_is_calibration_sup_job(tags):
    # The third tag, if present and set to True, indicates this was
    # requested by the calibration supervisor
    return len(tags) >= 3 and tags[2]

def get_metainfo(logfile: Labber.LogFile):
    tags = extract_tags(logfile)
    return (get_job_id(tags), get_script_name(tags), get_is_calibration_sup_job(tags))

# =========================================================================
# BCC / MSS updating
# =========================================================================

def update_mss_and_bcc(memory, job_id):

    # Helper printout with first 5 outcomes
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
        str(BCC_MACHINE_ROOT_URL) + REST_API_MAP["logfiles"] + "/" + job_id  # correct?
    )
    print(f"Download url: {download_url}")
    response = requests.put(
        MSS_JOB + REST_API_MAP["download_url"], json=download_url
    )
    if response:
        print("Updated job download_url on MSS")


# Running the postprocessing_worker from the command-line for testing purposes
# Note: files with missing tags may not work
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocessing stand-alone program")
    parser.add_argument("--logfile", "-f", default="", type=str)
    args = parser.parse_args()

    logfile = args.logfile

    results = postprocess(logfile)

    print(f"{results=}")
