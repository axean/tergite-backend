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

import logging
import math
from numbers import Number
from random import random
from typing import Dict, List, Union

import numpy as np
import redis
import toml
from scipy.optimize import curve_fit

from backend_properties_config.initialize_properties import get_component_ids
from calibration.calibration_common import (
    DataStatus,
    JobDoneEvent,
    get_post_processed_result,
    read_calibration_result,
    request_job,
    write_calibration_result,
)
from measurement_jobs.measurement_jobs import (
    mk_job_calibrate_signal_demodulation,
    mk_job_check_signal_demodulation,
    mk_job_pulsed_resonator_spectroscopy,
    mk_job_rabi,
    mk_job_ramsey,
    mk_job_two_tone,
    mk_job_vna_resonator_spectroscopy,
)

# Initialize logger

logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
# The following two lines are not used yet, but can be good to have available:
logger.setLevel(logging.INFO)
LOGLEVEL = logging.INFO

# Set up Redis connection
red = redis.Redis(decode_responses=True)


# Type aliases
JobID = str

# -------------------------------------------------------------------------
# Check data procedures

# This function is just a template for a future implementation
# check_data will do something like this:
async def check_dummy(node: str, job_done_event: JobDoneEvent) -> DataStatus:
    # signal demodulation measurement is used as a dummy here.
    job = mk_job_check_signal_demodulation()

    job_id = job["job_id"]
    print(f"Requesting check job with {job_id=} for {node=} ...")
    await request_job(job, job_done_event)

    calibration_params = red.lrange(f"{prefix}:goal_parameters:{node}", 0, -1)
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

    # TODO return status based on the above param checks instead of
    # deciding at random (will be done when check_data is implemented
    # for the respective function)
    num = random()
    if num < 0.8:
        print(f"Check_data for {node} gives IN_SPEC")
        return DataStatus.in_spec
    if num < 0.95:  # remove this later :-)
        print(f"Check_data for {node} gives OUT_OF_SPEC")
        return DataStatus.out_of_spec
    print(f"Check_data for {node} gives BAD_DATA")
    return DataStatus.bad_data


# -------------------------------------------------------------------------
# Calibration procedures


async def calibrate_vna_resonator_spectroscopy(node: str, job_done_event: JobDoneEvent):
    """VNA resonator spectroscopy (for Labber compatible equipment)"""

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

    resonators = get_component_ids("resonator")

    # The following values are better than the design values. The
    # problem (LokeB 2023-03-29) with the "design" values is that they
    # differ more from the actual values than the difference between
    # the resonators' actual values. Currently we don't have any
    # obvious method to automatically find out which measurement
    # results belong to "our" resonators and not, given only the
    # design values. The "expected" values are used instead, to
    # demonstrate the concept.
    resonator_expected_frequencies = measurement_config[
        "resonator_expected_frequencies"
    ]
    # The component labels in the local measurement TOML file should
    # correspond to system configured component ids
    _assert_same_component_ids(
        "resonator", list(resonator_expected_frequencies.keys()), resonators
    )

    # -------------------------------------------------------------------------
    # Initialize some meta-information (only used in a "notes" field below)
    local_sweep_powers = _get_powers(local_sweep_power)
    local_sweep_middle_index = len(local_sweep_powers) // 2
    local_sweep_median_power = local_sweep_powers[local_sweep_middle_index]

    # -------------------------------------------------------------------------
    # Global frequency sweep to find the resonators

    job = mk_job_vna_resonator_spectroscopy(
        freq_start=big_sweep_start,
        freq_stop=big_sweep_stop,
        num_pts=big_sweep_num_pts,
        # N.B. Called num_ave for VNA RS and qa_avg for pulsed
        # spectroscopy
        num_ave=big_sweep_num_ave,
        power=big_sweep_power,
    )

    job_id = job["job_id"]

    print(f"Requesting calibration job with {job_id=} for {node=} ...")
    await request_job(job, job_done_event)

    # Post-processed results are now available via job_id
    result_big_sweep = get_post_processed_result(job_id)

    [low_power_sweep, high_power_sweep] = result_big_sweep
    if min(len(low_power_sweep), len(high_power_sweep)) < len(resonators):
        message = f"Critical error: too few fits found: low power sweep:{len(low_power_sweep)}, high power sweep:{len(high_power_sweep)}"
        logger.error(message)
        # TODO: better error handling
        exit(1)
    if len(low_power_sweep) != len(high_power_sweep):
        message = f"Critical error:  low and high power sweeps have different number of fits: low power sweep:{len(low_power_sweep)}, high power sweep:{len(high_power_sweep)}"
        logger.error(message)
        # TODO: better error handling
        exit(1)

    """
    Resonance frequencies: (example from LokeB, 2023-03-07)
    [[6331400000.0, 6415800000.0, 6684500000.0, 6759800000.0, 6986700000.0],
     [6330700000.0, 6415900000.0, 6683900000.0, 6759100000.0, 6985900000.0]]
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
        "Big sweep frequency shifts for the negative peaks "
        f"in the order they were found:{enumerated_frequency_shifts}"
    )

    # Check which frequency shifts look OK
    for i, shift in enumerated_frequency_shifts:
        if shift >= min_frequency_diff:
            print(f"Qubit {i} OK for fit {i}")
        else:
            print(
                f"Qubit {i} NOT OK for fit {i}, {shift=}, "
                f"but should be {min_frequency_diff}"
            )

    # -------------------------------------------------------------------------
    # Local sweep for each big sweep fit, for closer frequency estimates
    # -------------------------------------------------------------------------

    local_start_frequency = lambda i: result_big_sweep[1][i] - delta_span
    local_stop_frequency = lambda i: result_big_sweep[0][i] + delta_span

    local_sweep_jobs = [
        mk_job_vna_resonator_spectroscopy(
            freq_start=local_start_frequency(i),
            freq_stop=local_stop_frequency(i),
            num_pts=local_sweep_num_pts,
            power=local_sweep_power,
            # Note that we use a "custom" post-processing function:
            post_processing="process_resonator_spectroscopy_vna_phase_2",
        )
        # index the jobs as the fits we found in the big sweep
        for i in range(len(result_big_sweep[0]))
    ]

    # We don't know yet which fits actually correspond to "our"
    # resonators, but we measure them all anyway, to have the results
    # reported in the terminal:
    median_power_results = {}  # collect results
    for i, job in enumerate(local_sweep_jobs):
        job_id = job["job_id"]
        await request_job(job, job_done_event)
        # post-processed results are now available via job_id
        result_local_sweep = get_post_processed_result(job_id)

        # Local sweep frequency shift (a.k.a. "Chi_01"):
        # Resonance at low power - resonance at high power
        chi_01 = result_local_sweep[0]["fr"] - result_local_sweep[2]["fr"]

        logger.info(
            f"Local sweep for fit {i}, "
            f"interval=[{local_start_frequency(i)}, "
            f"{local_stop_frequency(i)}], "
            f"resonance at {local_sweep_power[1]} dBm: "
            f"{result_local_sweep[2]['fr']} Hz, "
            f"resonance at {local_sweep_power[0]} dBm: "
            f"{result_local_sweep[0]['fr']} Hz, "
            f"chi_01 (local sweep frequency shift): {chi_01} Hz."
        )

        # Select the median power as a good guess, if defined.
        # result_local_sweep comes from postprocessing_worker,
        # process_resonator_spectroscopy_vna_phase_2:
        # position 0: min power, 1: median of powers, 2: max power
        median_power_results[i] = result_local_sweep[1]["fr"]

    logger.debug(f"{median_power_results=}")
    # Choose the values closest to the configured "expected" values (see above)
    for id in resonators:
        expected_frequency = resonator_expected_frequencies[id]
        diffs = [
            (abs(expected_frequency - fr), i, fr)
            for (i, fr) in median_power_results.items()
        ]
        logger.debug("{diffs=}")

        (_, i, frequency) = min(diffs)
        logger.info(
            f"For resonator {id}, fit {i=} {frequency=} was closest to "
            f"{expected_frequency=}, "
        )
        write_calibration_result(
            node,
            property_name="resonant_frequency",
            value=frequency,
            component="resonator",
            component_id=id,
            notes=f"VNA resonator spectroscopy: resonant frequency at "
            f"{local_sweep_median_power} dBm",
        )
        write_calibration_result(
            node,
            property_name="frequency_shift",
            value=frequency_shifts[i],
            component="resonator",
            component_id=id,
            notes=f"VNA resonator spectroscopy: big sweep frequency shift",
        )


async def calibrate_pulsed_resonator_spectroscopy(
    node: str, job_done_event: JobDoneEvent
):
    """Pulsed resonator spectroscopy using Zürich Instruments/Labber

    For each resonator, perform pulsed resonator spectroscopy in a
    small intervals near previously approximated resonant frequencies,
    e.g, obtained from VNA resonator spectroscopy, given in this
    routine's associated TOML file.
    """

    # -------------------------------------------------------------------------
    # Read parameters from TOML file, specific for this measurement
    # routine (calibrate_pulsed_resonator_spectroscopy)

    measurement_config = toml.load("calibration/pulsed_resonator_spectroscopy.toml")
    common_measurement_parameters = measurement_config["common_measurement_parameters"]
    resonator_measurement_parameters = measurement_config[
        "resonator_measurement_parameters"
    ]

    resonators = get_component_ids("resonator")
    # The component labels in the local measurement TOML file should
    # correspond to system configured component ids
    _assert_same_component_ids(
        "resonator", list(resonator_measurement_parameters.keys()), resonators
    )

    # LO sweep interval size / 2
    delta = measurement_config["readout_frequency_lo_delta"]
    # Low-band part of the readout frequency
    qa_if_limit = measurement_config["qa_if_limit"]

    vna_results = {}
    for id in resonators:
        value = resonator_measurement_parameters[id]["vna_resonant_frequency"]
        # Note (*):
        # The IF parts will all be the same in the current
        # implementation, but if we for some reason want to change the
        # way we split the RF into LO and IF (for instance retaining
        # the decimal fraction of the RF value in the IF part), it
        # will be encapsulated in the splitting function. Therefore we
        # will retrieve the IF parts from this list even if the IF
        # value from the configuration file happens to be known here.
        vna_results[id] = _split(value, qa_if_limit)

    results = {}
    for id in resonators:
        readout_frequency_lo = vna_results[id]["lo"]
        readout_frequency_lo_start = readout_frequency_lo - delta
        readout_frequency_lo_stop = readout_frequency_lo + delta
        job = mk_job_pulsed_resonator_spectroscopy(
            **common_measurement_parameters,
            readout_frequency_lo_start=readout_frequency_lo_start,
            readout_frequency_lo_stop=readout_frequency_lo_stop,
            readout_frequency_if=vna_results[id]["if"],
        )
        job_id = job["job_id"]
        logger.info(f"Performing {node} calibration, resonator id={id}, {job=}")

        await request_job(job, job_done_event)

        # post-processed results are now available via job_id
        result = get_post_processed_result(job_id)
        # The post-processed result is a dict in a singleton list,
        # therefore index 0:
        results[id] = result[0]["fr"]

    logger.info(f"Measurement results for {node}: {resonators=}, {results=}")

    # Save results in Redis:
    for id in resonators:
        # Put together the LO (results) and IF frequencies
        value = results[id] + vna_results[id]["if"]
        # Save it for internal use *and* publish it externally:
        write_calibration_result(
            node,
            property_name="resonant_frequency",
            value=value,
            component="resonator",
            component_id=id,
            notes=f"Pulsed resonator spectroscopy for {id}",
        )


async def calibrate_two_tone(node: str, job_done_event: JobDoneEvent):
    """Pulsed two-tone spectroscopy using Zürich Instruments/Labber"""

    # -------------------------------------------------------------------------
    # Read parameters from TOML file, specific for this measurement
    # routine (calibrate_two_tone)

    measurement_config = toml.load("calibration/two_tone.toml")
    common_measurement_parameters = measurement_config["common_measurement_parameters"]
    qubit_measurement_parameters = measurement_config["qubit_measurement_parameters"]

    qubits = get_component_ids("qubit")
    # The component labels in the local measurement TOML file should
    # correspond to system configured component ids
    _assert_same_component_ids(
        "qubit", list(qubit_measurement_parameters.keys()), qubits
    )
    # Note: (**)
    # We assume resonators and qubits have aligned ids, in the sense
    # that the sorted list of resonator ids correspond to the same
    # transmons as the sorted list of resonator ids.  This may need to
    # be addressed better in future versions. This should be known
    # after loading the device configuration.
    resonators = get_component_ids("resonator")

    qa_if_limit = measurement_config["qa_if_limit"]

    pulsed_results = _get_results_pulsed_resonator_spectroscopy(qa_if_limit)

    results = {}
    for id, r_id in zip(qubits, resonators):
        job = mk_job_two_tone(
            **common_measurement_parameters,
            **qubit_measurement_parameters[id],
            # See note (*) in calibrate_pulsed_resonator_spectroscopy
            readout_frequency_if=pulsed_results[r_id]["if"],
            readout_frequency_lo=pulsed_results[r_id]["lo"],
        )
        job_id = job["job_id"]
        logger.info(f"Performing {node} calibration, qubit id={id}, {job=}")

        await request_job(job, job_done_event)

        # post-processed results are now available via job_id
        result = get_post_processed_result(job_id)
        # The post-processed result is a singleton list, therefore index 0:
        results[id] = result[0]

    logger.info(f"Measurement results for {node}: {qubits=}, {results=}")

    # Save in Redis:
    for id in qubits:
        # add lower and higher parts:
        value = results[id] + qubit_measurement_parameters[id]["drive_frequency_lo"]
        write_calibration_result(
            node,
            property_name="excitation_frequency",
            value=value,
            component="qubit",
            component_id=id,
            # NOTE: If we don't want to publish this for external use, set:
            # publish=False,
            notes=f"Pulsed two-tone qubit spectroscopy for {id}",
        )


async def calibrate_rabi(node: str, job_done_event: JobDoneEvent):
    """Pulsed Rabi spectroscopy using Zürich Instruments/Labber"""

    # -------------------------------------------------------------------------
    # Read parameters from TOML file, specific for this measurement
    # routine (calibrate_rabi)

    measurement_config = toml.load("calibration/rabi.toml")
    common_measurement_parameters = measurement_config["common_measurement_parameters"]
    qubit_measurement_parameters = measurement_config["qubit_measurement_parameters"]

    # We will need a few parameters from the two_tone config later:
    measurement_config_two_tone = toml.load("calibration/two_tone.toml")
    two_tone_qubit_measurement_parameters = measurement_config_two_tone[
        "qubit_measurement_parameters"
    ]

    qubits = get_component_ids("qubit")
    # The component labels in the local measurement TOML file should
    # correspond to system configured component ids
    _assert_same_component_ids(
        "qubit", list(qubit_measurement_parameters.keys()), qubits
    )

    # See note (**) in calibrate_two_tone
    resonators = get_component_ids("resonator")

    qa_if_limit = measurement_config["qa_if_limit"]

    pulsed_results = _get_results_pulsed_resonator_spectroscopy(qa_if_limit)

    two_tone_results = _get_results_two_tone(two_tone_qubit_measurement_parameters)

    results = {}
    for id, r_id in zip(qubits, resonators):
        job = mk_job_rabi(
            **common_measurement_parameters,
            **qubit_measurement_parameters[id],
            readout_frequency_if=pulsed_results[r_id]["if"],
            readout_frequency_lo=pulsed_results[r_id]["lo"],
            drive_frequency_if=two_tone_results[id]["if"],
            drive_frequency_lo=two_tone_results[id]["lo"],
        )
        job_id = job["job_id"]
        logger.info(f"Performing {node} calibration, qubit id={id}, {job=}")

        await request_job(job, job_done_event)

        # post-processed results are now available via job_id
        result = get_post_processed_result(job_id)
        # The post-processed result is a singleton list, therefore index 0:
        results[id] = result[0]

    logger.info(f"Measurement results for {node}: {qubits=}, {results=}")

    # Save in Redis:
    for id in qubits:
        value = results[id]
        write_calibration_result(
            node,
            property_name="pi_pulse_amplitude",
            value=value,
            component="qubit",
            component_id=id,
            notes=f"Rabi pi-pulse amplitude for {id}",
        )


async def calibrate_ramsey(node: str, job_done_event: JobDoneEvent):
    """Pulsed Ramsey spectroscopy using Zürich Instruments/Labber"""

    # -------------------------------------------------------------------------
    # Read parameters from TOML file, specific for this measurement
    # routine (calibrate_ramsey)

    measurement_config = toml.load("calibration/ramsey.toml")
    common_measurement_parameters = measurement_config["common_measurement_parameters"]
    qubit_measurement_parameters = measurement_config["qubit_measurement_parameters"]

    # We will need a few parameters from the two_tone config later:
    measurement_config_two_tone = toml.load("calibration/two_tone.toml")
    two_tone_qubit_measurement_parameters = measurement_config_two_tone[
        "qubit_measurement_parameters"
    ]

    qubits = get_component_ids("qubit")
    # The component labels in the local measurement TOML file should
    # correspond to system configured component ids
    _assert_same_component_ids(
        "qubit", list(qubit_measurement_parameters.keys()), qubits
    )

    # See note (**) in calibrate_two_tone
    resonators = get_component_ids("resonator")

    qa_if_limit = measurement_config["qa_if_limit"]

    pulsed_results = _get_results_pulsed_resonator_spectroscopy(qa_if_limit)

    two_tone_results = _get_results_two_tone(two_tone_qubit_measurement_parameters)

    rabi_results = _get_results_rabi()

    offsets = measurement_config["offsets"]

    qubit_offset_results = {}
    for id, r_id in zip(qubits, resonators):
        offset_results = {}
        # Perform a measurement for each offset, added to the two_tone result
        for offset in offsets:
            job = mk_job_ramsey(
                **common_measurement_parameters,
                **qubit_measurement_parameters[id],
                readout_frequency_if=pulsed_results[r_id]["if"],
                readout_frequency_lo=pulsed_results[r_id]["lo"],
                drive_frequency_if=two_tone_results[id]["if"] + offset,
                drive_frequency_lo=two_tone_results[id]["lo"],
                drive_amp=rabi_results[id] / 2,
            )
            job_id = job["job_id"]
            logger.info(f"Performing {node} calibration, qubit id={id}, {job=}")
            await request_job(job, job_done_event)

            # post-processed results are now available via job_id
            result = get_post_processed_result(job_id)
            # The post-processed result is a singleton list, therefore index 0:
            correction_value = result[0]
            # For positive offests, the correction is negative, and
            # for negative offsets, the correction is positive:
            offset_results[offset] = -math.copysign(correction_value, offset)
        qubit_offset_results[id] = offset_results

    logger.info(f"Measurement results for {node}: {qubits=}, {qubit_offset_results=}")

    results = {}
    # For curve_fit to fit with offsets and respective qubit_offset_results:
    straight_line = lambda x, a, b: a * x + b
    for id in qubits:
        y_data = [qubit_offset_results[id][offset] for offset in offsets]
        _slope, y_intercept = curve_fit(straight_line, offsets, y_data)[0]
        results[id] = y_intercept

    logger.info(f"Straight line interpolated results for {node}: {qubits=}, {results=}")

    # Save results in Redis:
    for id in qubits:
        value = (
            # correction value
            results[id]
            # two_tone IF and LO parts
            + two_tone_results[id]["if"]
            + two_tone_results[id]["lo"]
        )

        write_calibration_result(
            node,
            property_name="excitation_frequency",
            value=value,
            component="qubit",
            component_id=id,
            notes=f"Pulsed Ramsey qubit spectroscopy for {id}",
        )


async def calibrate_dummy(node: str, job_done_event: JobDoneEvent):
    # Note: using this only works like a "demo", we are going to
    # refactor this later. Demodulation measurement is used as a dummy
    # here.
    job = mk_job_calibrate_signal_demodulation()

    job_id = job["job_id"]
    print(f"Requesting calibration job with {job_id=} for {node=} ...")
    await request_job(job, job_done_event)

    print("")

    calibration_params = red.lrange(f"{prefix}:goal_parameters:{node}", 0, -1)
    for calibration_param in calibration_params:

        # Fetch the values we got from the calibration's post-processing
        result_key = f"postprocessing:results:{job_id}"
        result = red.get(result_key)
        print(
            f"For {calibration_param=}, from Redis we read {result_key} from postprocessing: {result}"
        )
        if result == None:
            print(f"Warning: no entry found for key {result_key}")
            result = "not found"  # TODO: better error handling

        # TODO: This is behind bcc_dev2. The calibration_param is
        # associated with a component type that has a list of
        # component_id's. We should iterate over that and call
        # write_calibration_result, so it gets stored in Redis. since
        # this is just demo code, it isn't critical right now. The
        # effect of not having implemented this is that the
        # calibration will be re-run each time the main loop checks if
        # calibration is needed.


# -------------------------------------------------------------------------
# Misc helpers


def _get_powers(power_spec: Union[Number, List[Number]]) -> List[Number]:
    if isinstance(power_spec, Number):
        return [power_spec]
    else:  # expects a list of the form [min, max, step_size]
        return list(np.linspace((*tuple(power_spec))))


# See note (*) in calibrate_pulsed_resonator_spectroscopy
def _split(rf_frequency: float, if_frequency_limit: float) -> Dict[str, float]:
    return {"lo": rf_frequency - if_frequency_limit, "if": if_frequency_limit}


def _assert_same_component_ids(
    component: str, local_ids: List[str], system_ids: List[str]
):
    """Requires that the two string lists are equal, and exit
    otherwise. This is to assure that the component ids in the local
    measurement configuration file match the system configured
    component ids.
    """
    if system_ids != sorted(local_ids):  # the system_ids are already sorted
        logger.error(
            f"Local {component} measurement config ids {local_ids} "
            f"not consistent with system configured {component} ids {system_ids}",
            stacklevel=2,
        )
        # NOTE: alternatively we could require just that the toml ids
        # are a subset of the system configured ids. However,
        # currently we do have parameters for all ids, and if, by some
        # reason, we have an id without parameters, we can just have
        # an empty entry for it.
        exit(1)


def _get_results_pulsed_resonator_spectroscopy(
    qa_if_limit: float,
) -> Dict[str, Dict[str, float]]:
    resonators = get_component_ids("resonator")

    pulsed_results = {}
    for id in resonators:
        value, _timestamp = read_calibration_result(
            "pulsed_resonator_spectroscopy",
            "resonant_frequency",
            component="resonator",
            component_id=id,
        )
        pulsed_results[id] = _split(value, qa_if_limit)

    return pulsed_results


def _get_results_two_tone(
    qubit_measurement_parameters: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    qubits = get_component_ids("qubit")

    two_tone_results = {}
    for id in qubits:
        value, _timestamp = read_calibration_result(
            "two_tone",
            "excitation_frequency",
            component="qubit",
            component_id=id,
        )
        drive_frequency_lo = qubit_measurement_parameters[id]["drive_frequency_lo"]
        drive_frequency_if = value - drive_frequency_lo
        two_tone_results[id] = _split(value, drive_frequency_if)

    return two_tone_results


def _get_results_rabi() -> Dict[str, float]:
    qubits = get_component_ids("qubit")

    rabi_results = {}
    for id in qubits:
        value, _timestamp = read_calibration_result(
            "rabi",
            "pi_pulse_amplitude",
            component="qubit",
            component_id=id,
        )
        rabi_results[id] = value  # these are amplitudes, and are not split

    return rabi_results
