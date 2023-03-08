# This code is part of Tergite
#
# (C) Johan Blomberg, Gustav Grännsjö 2020
# (C) Copyright Miroslav Dobsicek 2020, 2021
# (C) Copyright David Wahlstedt 2022, 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import ast
import json
import logging
from numbers import Number
from pathlib import Path
from random import random
from tempfile import gettempdir
from typing import Any, List, Union
from uuid import uuid4

import numpy as np
import redis
import requests
import toml

import settings
from backend_properties_config.initialize_properties import get_n_components
from calibration.calibration_common import DataStatus, write_calibration_goal
from measurement_jobs.measurement_jobs import (
    mk_job_calibrate_signal_demodulation,
    mk_job_check_signal_demodulation,
    mk_job_vna_resonator_spectroscopy,
)
from utils import datetime_utils

# Initialize logger

logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
# The following two lines are not used yet, but can be good to have available:
logger.setLevel(logging.INFO)
LOGLEVEL = logging.INFO

# Set up Redis connection
red = redis.Redis(decode_responses=True)

# Settings
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL

REST_API_MAP = {"jobs": "/jobs"}

# Type aliases
JobID = str

# -------------------------------------------------------------------------
# Check data procedures

# This function is just a template for a future implementation
# check_data will do something like this:
async def check_dummy(node, job_done_event) -> DataStatus:
    # signal demodulation measurement is used as a dummy here.
    job = mk_job_check_signal_demodulation()

    job_id = job["job_id"]
    print(f"Requesting check job with {job_id=} for {node=} ...")
    await request_job(job, job_done_event)

    calibration_params = red.lrange(f"m_params:{node}", 0, -1)
    for calibration_param in calibration_params:
        # Fetch the values we got from the measurement's post-processing
        # here you can use the calibration_param
        result_key = f"postprocessing:results:{job_id}"
        result = red.get(result_key)
        print(
            f"check_data: For {calibration_param=}, from Redis we read {result_key} from postprocessing: {result}"
        )
        if result == None:
            print(f"Warning: no entry found for key {result_key}")
        # TODO ensure value is within thresholds

    # TODO return status based on the above param checks instead of deciding at random
    num = random()
    if num < 0.8:
        print(f"Check_data for {node} gives IN_SPEC")
        return DataStatus.in_spec
    if num < 0.95:  # remove this later :-)
        print(f"Check_data for {node} gives OUT_OF_SPEC")
        return DataStatus.out_of_spec
    print(f"Check_data for {node} gives BAD_DATA")
    return DataStatus.bad_data


# To be used by calibration nodes that don't yet have check_data implemented
async def check_return_out_of_spec(node, _job_done_event):
    print(f"check_data not implemented for {node}, forcing calibration ...")
    return DataStatus.out_of_spec


# -------------------------------------------------------------------------
# Calibration procedures


async def calibrate_vna_resonator_spectroscopy(node, job_done_event):
    """VNA resonator spectroscopy (for Labber compatible equipment)"""

    # -------------------------------------------------------------------------
    # Read some configuration from Redis

    n_resonators = get_n_components("resonator")

    # -------------------------------------------------------------------------
    # Read parameters from TOML file, specific for this measurement
    # routine (calibrate_vna_resonator_spectroscopy)

    measurement_config = toml.load("calibration/vna_resonator_spectroscopy.toml")

    big_sweep_start = measurement_config["big_sweep_start"]
    big_sweep_stop = measurement_config["big_sweep_stop"]

    big_sweep_num_pts = measurement_config["big_sweep_num_pts"]
    big_sweep_num_ave = measurement_config["big_sweep_num_ave"]
    big_sweep_power = measurement_config["big_sweep_power"]

    local_sweep_num_pts = measurement_config["local_sweep_num_pts"]
    local_sweep_power = measurement_config["local_sweep_power"]

    min_frequency_diff = measurement_config["min_frequency_diff"]
    delta_span = measurement_config["delta_span"]

    design_resonator_frequencies=measurement_config["design_resonator_frequencies"]

    # For set_resonator_property operation
    source = "measurement"

    # -------------------------------------------------------------------------
    # Initialize some meta-information
    local_sweep_powers = _get_powers(local_sweep_power)
    local_sweep_middle_index = len(local_sweep_powers) // 2
    local_sweep_median_power = local_sweep_powers[local_sweep_middle_index]

    # -------------------------------------------------------------------------
    # Global frequency sweep to find the resonators

    job = mk_job_vna_resonator_spectroscopy(
        freq_start=big_sweep_start,
        freq_stop=big_sweep_stop,
        num_pts=big_sweep_num_pts,
        num_ave=big_sweep_num_ave,
        power=big_sweep_power,
    )

    job_id = job["job_id"]

    print(f"Requesting calibration job with {job_id=} for {node=} ...")
    await request_job(job, job_done_event)

    # post-processed results are now available via job_id
    result_big_sweep = get_post_processed_result(job_id)

    # post-processed results are now successfully extracted into result_big_sweep
    [low_power_sweep, high_power_sweep] = result_big_sweep
    if min(len(low_power_sweep), len(high_power_sweep)) < n_resonators:
        message = f"Critical error: too few resonators found: low power sweep:{len(low_power_sweep)}, high power sweep:{len(high_power_sweep)}"
        print(message)
        # what to do?  This kind of failure is above the job level, so
        # it shouldn't be the job supervisor, rather the calibration
        # supervisor. Maybe we could have a property for the resonator,
        # if it's not responding at some frequency.
        return

    """
    Resonance frequencies: (example from LokeB, 2023-03-07)
    [[6331400000.0, 6415800000.0, 6684500000.0, 6759800000.0, 6986700000.0],
     [6330700000.0, 6415900000.0, 6683900000.0, 6759100000.0, 6985900000.0]]
    For now, ignore which resonators are "right" and not: just scan
    them, an we see in phase 2 how they look
    """

    # Resonance frequency at low power > resonance frequency at high power
    # therefore, high_resonance_frequency > low_resonance_frequency below:
    frequency_shifts = [
        high_resonance_frequency - low_resonance_frequency
        for (high_resonance_frequency, low_resonance_frequency) in zip(
            low_power_sweep, high_power_sweep
        )
    ]
    enumerated_frequency_shifts = list(enumerate(frequency_shifts))
    logger.info(
        f"Big sweep frequency shifts for the resonators:{enumerated_frequency_shifts}"
    )

    # Populate backend properties with frequency shifts
    for i, shift in enumerated_frequency_shifts:
        # NOTE:
        # Alternatively, one could recast this property into a qubit
        # property "functional", with a boolean value, True if and only if
        # frequency_shift is above specified level. Would that be better?
        #
        # As mentioned before, one could also add a note about at what
        # powers this was measured.
        write_calibration_goal(
            node,
            property_name="frequency_shift",
            value=shift,
            component="resonator",
            index=i,
        )
        if shift >= min_frequency_diff:
            print(f"Qubit {i} OK for resonator {i}")
        else:
            print(
                f"Qubit {i} NOT OK for resonator {i}, {shift=}, "
                f"but should be {min_frequency_diff}"
            )

    # -------------------------------------------------------------------------
    # Local sweep for each resonator, for closer resonance frequency estimates
    # -------------------------------------------------------------------------

    local_start_frequency = lambda i: result_big_sweep[1][i] - delta_span
    local_stop_frequency = lambda i: result_big_sweep[0][i] + delta_span

    local_sweep_jobs = [
        mk_job_vna_resonator_spectroscopy(
            freq_start=local_start_frequency(i),
            freq_stop=local_stop_frequency(i),
            num_pts=local_sweep_num_pts,
            power=local_sweep_power,
            post_processing="process_resonator_spectroscopy_vna_phase_2",
        )
        for i in range(n_resonators)
    ]

    for i, job in enumerate(local_sweep_jobs):
        job_id = job["job_id"]
        await request_job(job, job_done_event)
        # post-processed results are now available via job_id
        result_local_sweep = get_post_processed_result(job_id)

        # Local sweep frequency shift (a.k.a. "Chi_01"):
        # Resonance at low power - resonance at high power
        chi_01 = result_local_sweep[0]["fr"] - result_local_sweep[2]["fr"]

        logger.info(
            f"Local sweep for resonator {i}, "
            f"interval=[{local_start_frequency(i)}, "
            f"{local_stop_frequency(i)}], "
            f"resonance at {local_sweep_power[1]} dBm: "
            f"{result_local_sweep[2]['fr']} Hz, "
            f"resonance at {local_sweep_power[0]} dBm: "
            f"{result_local_sweep[0]['fr']} Hz, "
            f"chi_01 (local sweep frequency shift): {chi_01} Hz."
        )

        # Select the median power as a good guess, if defined.
        write_calibration_goal(
            node,
            property_name="resonant_frequency",
            # result_local_sweep comes from postprocessing_worker,
            # process_resonator_spectroscopy_vna_phase_2:
            # position 0: min power, 1: median of powers, 2: max power
            value=result_local_sweep[1]["fr"],
            component="resonator",
            index=i,
            # This note could perhaps be added before the
            # measurements start, but then we need a way to know the
            # power from there. This is a simple solution for now, to
            # see what you think. We could as well just remove it.
            notes=f"VNA resonator spectroscopy: resonant frequency at "
            f"{local_sweep_median_power} dBm",
        )


async def calibrate_dummy(node, job_done_event):
    # Note: using this only works like a "demo", we are going to
    # refator this later. Demodulation measurement is used as a dummy
    # here.
    job = mk_job_calibrate_signal_demodulation()

    job_id = job["job_id"]
    print(f"Requesting calibration job with {job_id=} for {node=} ...")
    await request_job(job, job_done_event)

    print("")

    calibration_params = red.lrange(f"m_params:{node}", 0, -1)
    for calibration_param in calibration_params:
        # Note: unit is from now on stored elsewhere. See
        # initialize_properties.py
        unit = red.hget(f"m_params:{node}:{calibration_param}", "unit")

        # Fetch the values we got from the calibration's post-processing
        result_key = f"postprocessing:results:{job_id}"
        result = red.get(result_key)
        print(
            f"For {calibration_param=}, from Redis we read {result_key} from postprocessing: {result}"
        )
        if result == None:
            print(f"Warning: no entry found for key {result_key}")
            result = "not found"  # TODO: better error handling

        red.hset(f"param:{calibration_param}", "name", calibration_param)
        red.hset(f"param:{calibration_param}", "date", datetime_utils.utc_now_iso())
        red.hset(f"param:{calibration_param}", "unit", unit or "")
        red.hset(f"param:{calibration_param}", "value", result)


# -------------------------------------------------------------------------
# Misc helpers


async def request_job(job, job_done_event):
    job_id = job["job_id"]

    # Updating for handle_message to accept only this job_id:
    job_done_event.requested_job_id = job_id

    tmpdir = gettempdir()
    file = Path(tmpdir) / str(uuid4())
    with file.open("w") as dest:
        json.dump(job, dest)

    with file.open("r") as src:
        files = {"upload_file": src}
        url = str(BCC_MACHINE_ROOT_URL) + REST_API_MAP["jobs"]
        response = requests.post(url, files=files)

        # Right now the Labber Connector sends a response *after*
        # executing the scenario i.e., the POST request is *blocking*
        # until after the measurement execution this will change in
        # the future; it should just ack a successful upload of a
        # scenario and nothing more

        if response:
            file.unlink()
            print("Job has been successfully sent")
        else:
            print("request_job failed")
            return

    # Wait until reply arrives(the one with our job_id).
    await job_done_event.event.wait()
    job_done_event.event.clear()

    print("")


def get_post_processed_result(job_id: JobID) -> Any:
    result_key = f"postprocessing:results:{job_id}"
    try:
        result_repr = red.get(result_key)
        result = ast.literal_eval(result_repr)
    except Exception as err:
        message = (
            f"Failed to obtain post-processed results from key {result_key}, {err=}"
        )
        print(message)
        inform_failure(job_id, message)
        return
    return result


def _get_powers(power_spec: Union[Number, List[Number]]) -> List[Number]:
    if isinstance(power_spec, Number):
        return [power_spec]
    else:  # expects a list of the form [min, max, step_size]
        return list(np.linspace((*tuple(power_spec))))
