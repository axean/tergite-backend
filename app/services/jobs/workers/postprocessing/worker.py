# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020, 2021
# (C) Copyright David Wahlstedt 2021, 2022, 2023
# (C) Copyright Abdullah Al Amin 2021, 2022
# (C) Copyright Axel Andersson 2022
# (C) Andreas Bengtsson 2020
# (C) Martin Ahindura 2023
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
import functools
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import redis
import requests
import tqcsf.file
from requests import Response
from sklearn.utils.extmath import safe_sparse_dot
from syncer import sync

import Labber
import settings
from app.utils import date_time
from app.utils.http import get_mss_client

from .....utils.representation import to_string
from ...service import (
    JobNotFound,
    Location,
    cancel_job,
    fetch_job,
    fetch_redis_entry,
    inform_failure,
    inform_location,
    register_job,
    save_result,
    update_final_location_timestamp,
    update_job_entry,
)
from .analysis import (
    find_resonators,
    fit_oscillation_itraces,
    fit_resonator_itraces,
    gaussian_fit_itraces,
)
from .dtos import LogfileType

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
    "timelog": "/timelog",
    "jobs": "/jobs",
    "logfiles": "/logfiles",
    "backends": "/backends",
}

# Type aliases

JobID = str

# Redis connection

red = redis.Redis(decode_responses=True)

# =========================================================================
# Post-processing entry function
# =========================================================================


def logfile_postprocess(
    logfile: Path, *, logfile_type: LogfileType = LogfileType.LABBER_LOGFILE
) -> JobID:
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

    # The return value will be passed to postprocessing_success_callback
    if logfile_type == LogfileType.TQC_STORAGE:
        print("Identified TQC storage file, reading file using tqcsf")
        sf = tqcsf.file.StorageFile(new_file, mode="r")
        return postprocess_tqcsf(sf)
    else:
        # Labber logfile
        # All further post-processing, from this point on, is Labber specific.
        labber_logfile = Labber.LogFile(new_file)
        return postprocess_labber_logfile(labber_logfile)


# =========================================================================
# Post-processing Quantify / Qblox files
# =========================================================================


# FIXME: This is a hardcoded solution for the eX3 demo on November 7, 2022
def _load_lda_discriminator(index: int) -> object:
    fn = f"state-disc-q{index}.disc"
    print("Loaded", fn, "for Loki discimination (2022-10-28)")
    with open(fn, mode="rb") as _file:
        lda_model = pickle.load(_file)
    return lda_model


# FIXME: This is a hardcoded solution for the eX3 demo on November 7, 2022
# TODO: Fetch discriminator from external source ?
def _hardcoded_discriminator(
    *, qubit_idx: int, iq_points: complex
) -> list:  # List[0/1]
    # _DISCRIMINATORS = {index: _load_lda_discriminator(index) for index in range(5)}
    _DISCRIMINATORS = {0: _load_lda_discriminator(0)}
    lda_model = _DISCRIMINATORS[qubit_idx]

    X = np.zeros((iq_points.shape[0], 2))
    X[:, 0] = iq_points.real
    X[:, 1] = iq_points.imag

    return lda_model.predict(X)


def _fetch_discriminator(
    lda_parameters: dict, qubit_idx: int, iq_points: npt.NDArray[np.complex128]
) -> npt.NDArray[np.int_]:
    # FIXME: This is currently only used in the simulator, but we have to find a way to use it everywhere

    level = "threeState" if settings.DISCRIMINATE_TWO_STATE else "twoState"
    coef = np.array(lda_parameters[f"q{qubit_idx}"][level]["coef"])
    intercept = np.array(lda_parameters[f"q{qubit_idx}"][level]["intercept"])

    X = np.zeros((iq_points.shape[0], 2))
    X[:, 0] = iq_points.real
    X[:, 1] = iq_points.imag

    scores = safe_sparse_dot(X, coef.T, dense_output=True) + intercept

    if settings.DISCRIMINATE_TWO_STATE:
        return scores.argmax(axis=1)
    else:
        return (scores.ravel() > 0).astype(np.int_)


def postprocess_tqcsf(sf: tqcsf.file.StorageFile) -> JobID:
    with get_mss_client() as mss_client:
        if sf.meas_level == tqcsf.file.MeasLvl.DISCRIMINATED:
            discriminator_fn = _hardcoded_discriminator

            if settings.FETCH_DISCRIMINATOR:
                backend = sf.header["qobj"]["backend"].attrs["backend_name"]
                discriminator_url = f'{MSS_MACHINE_ROOT_URL}{REST_API_MAP["backends"]}/{backend}/properties/lda_parameters'
                response = mss_client.get(discriminator_url)

                if response.status_code == 200:
                    discriminator_fn = functools.partial(
                        _fetch_discriminator, response.json()
                    )
                else:
                    print(f"Response error {response}")

            try:
                memory = sf.as_readout(
                    discriminator=discriminator_fn,
                    disc_two_state=settings.DISCRIMINATE_TWO_STATE,
                )
                save_result_in_mss_and_bcc(
                    mss_client=mss_client, memory=memory, job_id=sf.job_id
                )
            except Exception as exp:
                logging.error(exp)

        elif sf.meas_level == tqcsf.file.MeasLvl.INTEGRATED:
            save_result_in_mss_and_bcc(
                mss_client=mss_client, memory=[], job_id=sf.job_id
            )

        elif sf.meas_level == tqcsf.file.MeasLvl.RAW:
            save_result_in_mss_and_bcc(
                mss_client=mss_client, memory=[], job_id=sf.job_id
            )

        else:
            print("Warning: cannot postprocess invalid StorageFile.")

        # job["name"] was set to "pulse_schedule" when registered

    return sf.job_id


# =========================================================================
# Post-processing Labber logfiles
# =========================================================================


# =========================================================================
# Post-processing helpers in PROCESSING_METHODS
# labber_logfile: Labber.LogFile
# Dummy post-processing of signal demodulation
def process_demodulation(labber_logfile: Labber.LogFile) -> JobID:
    job_id = get_job_id_labber(labber_logfile)
    return job_id


# Qasm job example
def process_qiskit_qasm(labber_logfile: Labber.LogFile) -> str:
    job_id = get_job_id_labber(labber_logfile)

    # Extract System state
    memory = extract_system_state_as_hex(labber_logfile)

    with get_mss_client() as mss_client:
        save_result_in_mss_and_bcc(mss_client=mss_client, memory=memory, job_id=job_id)

    # DW: I guess something else should be returned? memory or parts of it?
    return job_id


# VNA resonator spectroscopy
def process_resonator_spectroscopy_vna_phase_1(
    labber_logfile: Labber.LogFile,
) -> List[List[float]]:
    return find_resonators(labber_logfile)


def process_resonator_spectroscopy_vna_phase_2(
    labber_logfile: Labber.LogFile,
) -> List[Dict[str, float]]:
    # The indices below represent the low power sweep, median power
    # sweep, and high power sweep, respectively. We pick the first,
    # middle(left middle if even length), and last trace of the logfile.
    n_traces = labber_logfile.getNumberOfEntries()
    return fit_resonator_itraces(labber_logfile, [0, n_traces // 2, n_traces - 1])


# (*) Note on pulsed resonator spectroscopy, two_tone, Rabi, and
#     Ramsey post-processing:
#
# Currently, this only supports one trace per measurement, and the
# index list of traces is [0] below. To adapt this to multiple
# resonators/qubits per measurement, we need to figure out whether we
# can have Labber to put all the relevant traces in one logfile, or if
# they would come in one logfile for each resonator/qubit. Based on
# this we will be able to select which trace incides will be passed to
# the analysis functions.


# Pulsed resonator spectroscopy
def process_pulsed_resonator_spectroscopy(
    labber_logfile: Labber.LogFile,
) -> List[Dict[str, float]]:
    return fit_resonator_itraces(labber_logfile, [0])


# Two-tone
def process_two_tone(labber_logfile: Labber.LogFile) -> List[float]:
    # fit qubit spectra
    return gaussian_fit_itraces(labber_logfile, [0])


# Rabi
def process_rabi(labber_logfile: Labber.LogFile) -> List[float]:
    # fit Rabi oscillation
    fits = fit_oscillation_itraces(labber_logfile, [0])
    return [entry["period"] for entry in fits]


# Ramsey
def process_ramsey(labber_logfile: Labber.LogFile) -> List[float]:
    # fit Ramsey oscillation
    fits = fit_oscillation_itraces(labber_logfile, [0])
    return [entry["freq"] for entry in fits]


# =========================================================================
# Post-processing function mappings

# Based on job["name"], these are the default post-processing methods
PROCESSING_METHODS = {
    # VNA resonator spectroscopy
    "resonator_spectroscopy": process_resonator_spectroscopy_vna_phase_1,
    # Four basic calibration steps
    "pulsed_resonator_spectroscopy": process_pulsed_resonator_spectroscopy,
    "pulsed_two_tone_qubit_spectroscopy": process_two_tone,
    "rabi_qubit_pi_pulse_estimation": process_rabi,
    "ramsey_qubit_freq_correction": process_ramsey,
    # Other
    "demodulation_scenario": process_demodulation,
    "qiskit_qasm_runner": process_qiskit_qasm,
    "qasm_dummy_job": process_qiskit_qasm,
}

# A map from strings to the corresponding function names
#
# If job["post_processing"] field has any of these values, it will
# override the PROCESSING_METHODS mapping
PROCESSING_STR_TO_FUNCTION = {
    # VNA resonator spectroscopy
    "process_resonator_spectroscopy_vna_phase_1": process_resonator_spectroscopy_vna_phase_1,
    "process_resonator_spectroscopy_vna_phase_2": process_resonator_spectroscopy_vna_phase_2,
    # Four basic calibration steps
    "process_pulsed_resonator_spectroscopy": process_pulsed_resonator_spectroscopy,
    "process_two_tone": process_two_tone,
    "process_rabi": process_rabi,
    "process_ramsey": process_ramsey,
    # Other
    "process_demodulation": process_demodulation,
    "process_qiskit_qasm": process_qiskit_qasm,
}

# =========================================================================
# Post-processing Labber logfiles


def postprocess_labber_logfile(labber_logfile: Labber.LogFile) -> JobID:
    job_id = get_job_id_labber(labber_logfile)
    (job_name, is_calibration_supervisor_job, post_processing) = get_metainfo(job_id)

    print(
        f"Entering postprocess_labber_logfile for script: {job_name}, {job_id=}, {is_calibration_supervisor_job=}"
    )

    postprocessing_fn = PROCESSING_METHODS.get(job_name)
    custom_postprocessing_fn = PROCESSING_STR_TO_FUNCTION.get(post_processing)

    if custom_postprocessing_fn:
        if not postprocessing_fn:
            print(
                f'Warning: no default post-processing matched job_name, but "post_processing" was specified as {post_processing}, and that one will be used.'
            )
        results = custom_postprocessing_fn(labber_logfile)
    elif postprocessing_fn:
        results = postprocessing_fn(labber_logfile)
    else:
        message = f"No post-processing method assigned, nor by {job_name=}, neither by {post_processing=}."
        print(message)
        inform_failure(job_id, message)
        return job_id

    # Post-processing was specified, either by job["name"], or by
    # job["post_processing"]
    red.set(f"postprocessing:results:{job_id}", to_string(results))
    return job_id


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


def postprocessing_success_callback(
    _rq_job, _rq_connection, result: JobID, *args, **kwargs
):
    # From logfile_postprocess:
    job_id = result
    inform_location(job_id, Location.FINAL_Q)
    update_final_location_timestamp(job_id, status="started")

    (script_name, is_calibration_supervisor_job, post_processing) = get_metainfo(job_id)

    status = fetch_job(job_id, "status")

    with get_mss_client() as mss_client:
        if status["failed"]["time"]:
            print(
                f"Job {job_id}, {script_name=}, {post_processing=} has failed: aborting. Status: {status}"
            )
            _update_location_timestamps_in_mss(mss_client=mss_client, job_id=job_id)
            return

        print(f"Job with ID {job_id}, {script_name=} has finished")
        if post_processing:
            print(
                f"Results post-processed by '{post_processing}' available by job_id in Redis."
            )
        if is_calibration_supervisor_job:
            print(f"Job was requested by calibration_supervisor: notifying caller.")
            sync(notify_job_done(job_id))

        inform_location(job_id, Location.FINAL_W)
        update_final_location_timestamp(job_id, status="finished")
        _update_location_timestamps_in_mss(mss_client=mss_client, job_id=job_id)


def postprocessing_failure_callback(
    _rq_job, _rq_connection, result: JobID, *args, **kwargs
):
    """Callback to be called when postprocessing fails"""
    with get_mss_client() as mss_client:
        _update_location_timestamps_in_mss(mss_client=mss_client, job_id=result)


# =========================================================================
# Labber logfile extraction helpers
# =========================================================================

Memory = Any  # TODO: change to correct type!


def extract_system_state_as_hex(logfile: Labber.LogFile) -> Memory:
    raw_data = logfile.getData("State Discriminator 2 States - System state")
    memory = []
    for entry in raw_data:
        memory.append([hex(int(x)) for x in entry])
    return memory


def extract_shots(logfile: Labber.LogFile) -> int:
    return int(logfile.getData("State Discriminator 2 States - Shots", 0)[0])


def extract_max_qubits(logfile: Labber.LogFile) -> int:
    return int(
        logfile.getData("State Discriminator 2 States - Max no. of qubits used", 0)[0]
    )


QobjID = str  # TODO: check that this type is correct!


def extract_qobj_id(logfile: Labber.LogFile) -> QobjID:
    return logfile.getChannelValue("State Discriminator 2 States - QObj ID")


def get_job_id_labber(labber_logfile: Labber.LogFile) -> JobID:
    tags = labber_logfile.getTags()
    if len(tags) == 0:
        # Print this message, then let it crash:
        print(f"Fatal: no tags in logfile. Can't extract job_id")
    return tags[0]


def get_metainfo(job_id: str) -> Tuple[str, str, str]:
    entry = fetch_redis_entry(job_id)
    script_name = entry["name"]
    is_calibration_supervisor_job = entry.get("is_calibration_supervisor_job", False)
    post_processing = entry.get("post_processing")
    return (script_name, is_calibration_supervisor_job, post_processing)


# =========================================================================
# BCC / MSS updating
# =========================================================================


def save_result_in_mss_and_bcc(mss_client: requests.Session, memory, job_id: JobID):
    """Updates both MSS and BCC with the memory part of the result

    Args:
        mss_client: the requests.Session that can query MSS
        memory: the memory part of the result, usually saved as {'result': {'memory': [...]}}
        job_id: the ID of the job
    """
    _debug_job_memory_list(memory)
    result = {"memory": memory}
    payload = {
        "result": result,
        "status": "DONE",
        "download_url": f'{BCC_MACHINE_ROOT_URL}{REST_API_MAP["logfiles"]}/{job_id}',
        "timelog.RESULT": date_time.utc_now_iso(),
    }

    # update BCC's redis database
    save_result(job_id, result)

    # update MSS
    response = _update_job_in_mss(mss_client=mss_client, job_id=job_id, payload=payload)
    if response:
        print("Pushed update to MSS")


def _update_location_timestamps_in_mss(mss_client: requests.Session, job_id: JobID):
    """Updates the job entry in MSS with the timestamps that have been saved for each pipeline location

    Args:
        mss_client: the requests.Session that can query MSS
        job_id: the ID of the job
    """
    entry = fetch_redis_entry(job_id)
    try:
        response = _update_job_in_mss(
            mss_client=mss_client, job_id=job_id, payload=entry["timestamps"]
        )
        if not response.ok:
            raise ValueError(
                f"failed to push timestamps for job {job_id}", response.text
            )
    except Exception as exp:
        logging.error(exp)
        raise exp


def _update_job_in_mss(
    mss_client: requests.Session, job_id: JobID, payload: dict
) -> Response:
    """Updates the job in MSS with the given payload

    Args:
        mss_client: the requests.Session that can query MSS
        job_id: the ID of the job
        payload: the new updates to apply to the given job in MSS

    Returns:
        the requests.Response received after request to MSS
    """
    mss_job_url = f'{MSS_MACHINE_ROOT_URL}{REST_API_MAP["jobs"]}/{job_id}'
    return mss_client.put(mss_job_url, json=payload)


def _debug_job_memory_list(memory: list):
    """
    Helper printout with first 5 outcomes
    """
    print("Measurement results:")
    for experiment_memory in memory:
        s = str(experiment_memory[:5])
        if experiment_memory[5:6]:
            s = s.replace("]", ", ...]")
        print(s)


# =========================================================================
# Running postprocessing_worker from command-line for testing purposes
# =========================================================================

# Note: files with missing tags may not work
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocessing stand-alone program")
    parser.add_argument("--logfile", "-f", default="", type=str)
    parser.add_argument("--name", "-n", default="test_name", type=str)
    parser.add_argument("--post_processing", "-p", default="", type=str)
    args = parser.parse_args()

    logfile = args.logfile
    post_processing = args.post_processing

    labber_logfile = Labber.LogFile(logfile)

    job_id = get_job_id_labber(labber_logfile)
    # check if job id already present
    try:
        _ = fetch_redis_entry(job_id)
        # job_id already in use, we can't proceed
        print(
            f"the logfile's job_id is already in use. please wait until it is finished or try a file with another job_id"
        )
        exit()
    except JobNotFound:
        pass  # this is the expected behaviour

    register_job(job_id)
    try:
        name = labber_logfile.getTags()[1]
    except IndexError:
        print(
            f"Name missing in logfile, using default '{args.name}' instead (or specify with -n)."
        )
        name = args.name

    update_job_entry(job_id, name, "name")
    update_job_entry(job_id, post_processing, "post_processing")

    return_value = postprocess_labber_logfile(labber_logfile)

    # This cleanup is omitted if the previous call raises an
    # exception, maybe we should fix that.
    cancel_job(job_id, "testing done")
    # Maybe we need a new job_supervisor method for this:
    red.hdel("job_supervisor", job_id)
    print(f"{return_value=}")
