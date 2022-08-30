# This code is part of Tergite
#
# (C) Copyright David Wahlstedt 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from uuid import uuid4

import numpy as np

# ------------------------------------------------------------------------------
# Calibration jobs

# Performs VNA resonator spectroscopy with ZI / Labber
def mk_job_res_spect_vna(
    # Mandatory parameters for measurement job
    f_start,
    f_stop,
    if_bw,
    num_pts,
    power,
    num_ave,
    # Optional arguments to override calibration supervisor defaults
    is_calibration_sup_job = True,
    name = "resonator_spectroscopy",
    # Optional arguments to override any other parameters from the
    # defaults TOML file in measurement_jobs/parameter_defaults/
    **kwargs,
):
    """
    Parameters
    ----------
        f_start   : (float) start sweep frequency [Hz]
        f_stop    : (float) stop sweep frequency [Hz]
        if_bw   : (float) IF bandwidth setting [Hz]
        num_pts : (int) number of frequency points
        power   : (int) output power of VNA [dBm]
        num_ave : (int) number of averages"""

    job = {
        "job_id": str(uuid4()),
        "type": "script",
        "is_calibration_sup_job": is_calibration_sup_job,
        "name": "resonator_spectroscopy",
        "params": {
            "f_start": f_start,
            "f_stop": f_stop,
            "if_bw": if_bw,
            "num_pts": num_pts,
            # For multiple power sweeps use: [start, stop, n_pts],
            # example:[-50, 0, 1]
            "power": power,
            "num_ave": num_ave,
            **kwargs,
        },
    }
    return job

# Performs pulsed resonator spectroscopy
def mk_job_pulsed_res_spect(
        # Mandatory parameters for measurement job
        num_pts,
        qa_avg,
        readout_amp,
        readout_start_freq,
        readout_stop_freq,
        readout_power,
        # Optional arguments to override calibration supervisor defaults
        is_calibration_sup_job = True,
        name = "pulsed_resonator_spectroscopy",
        # Optional arguments to override any other parameters from the
        # defaults TOML file in measurement_jobs/parameter_defaults/
        **kwargs,
):

    job = {
        "job_id": str(uuid4()),
        "type": "script",
        "is_calibration_sup_job": is_calibration_sup_job,
        "name": name,
        "params": {
            "num_pts": num_pts,  # 20M/200 points = 1k sweep frequency resolution
            "qa_avg": qa_avg,  # previously num_ave, 1024
            "readout_start_freq": readout_start_freq,
            "readout_stop_freq": readout_stop_freq,
            "readout_amp": readout_amp,
            "readout_power": readout_power,
            **kwargs,
        },
    }
    return job

def mk_job_two_tone(
    # Mandatory parameters for measurement job
    drive_start_freq,
    drive_stop_freq,
    readout_resonance_freq,  # depends on pulsed resonator spectroscopy result
    num_pts,
    # Optional arguments to override calibration supervisor defaults
    is_calibration_sup_job = True,
    name = "pulsed_two_tone_qubit_spectroscopy",
    # Optional arguments to override any other parameters from the
    # defaults TOML file in measurement_jobs/parameter_defaults/
    **kwargs,
):

    job = {
        "job_id": str(uuid4()),
        "type": "script",
        "is_calibration_sup_job": is_calibration_sup_job,
        "name": name,
        "params": {
            "drive_start_freq": drive_start_freq,
            "drive_stop_freq": drive_stop_freq,
            "readout_resonance_freq": readout_resonance_freq,
            "num_pts": num_pts,
            **kwargs,
        },
    }
    return job

def mk_job_rabi(
    # Mandatory parameters for measurement job
    readout_resonance_freq,  # depends on pulsed resonator spectroscopy result
    drive_freq,  # depends on two_tone
    num_pts, # maybe this doesn't need to be a parameter here?
    # Optional arguments to override calibration supervisor defaults
    is_calibration_sup_job = True,
    name = "rabi_qubit_pi_pulse_estimation",
    # Optional arguments to override any other parameters from the
    # defaults TOML file in measurement_jobs/parameter_defaults/
    **kwargs,
):
    job = {
        "job_id": str(uuid4()),
        "type": "script",
        "is_calibration_sup_job": is_calibration_sup_job,
        "name": name,
        "params": {
            "readout_resonance_freq": readout_resonance_freq,
            "drive_freq": drive_freq,
            "num_pts": num_pts,
            **kwargs,
        },
    }
    return job

def mk_job_ramsey(
    # Mandatory parameters for measurement job
    readout_resonance_freq,  # depends on pulsed resonator spectroscopy result
    drive_freq,  # depends on two_tone
    drive_amp,  # depends on rabi = result / 2
    num_pts, # maybe this doesn't need to be a parameter here?
    # Optional arguments to override calibration supervisor defaults
    is_calibration_sup_job = True,
    name = "ramsey_qubit_freq_correction",
    # Optional arguments to override any other parameters from the
    # defaults TOML file in measurement_jobs/parameter_defaults/
    **kwargs,
):
    job = {
        "job_id": str(uuid4()),
        "type": "script",
        "is_calibration_sup_job": is_calibration_sup_job,
        "name": name,
        "params": {
            "readout_resonance_freq": readout_resonance_freq,
            "drive_freq": drive_freq,
            "num_pts": num_pts,
            **kwargs,
        },
    }
    return job


# ------------------------------------------------------------------------------
# Misc jobs

# Signal demodulation, demo that performs dry-run on the Labber API,
# not involving any instruments
def mk_job_check_sig_demod():
    # here we should do something simpler than in the calibration fn
    signal_array = gen_array(["linspace", "0", "5", "5"])
    demod_array = gen_array(["geomspace", "1", "5", "4"])

    job = {
        "job_id": str(uuid4()),
        "type": "script",
        "is_calibration_sup_job": True,  # job requested by calibration framework
        "name": "demodulation_scenario",
        "params": {
            "Sine - Frequency": signal_array,
            "Demod - Modulation frequency": demod_array,
        },
    }

    return job


def mk_job_calibrate_sig_demod():
    signal_array = gen_array(["linspace", "0", "10", "5"])
    demod_array = gen_array(["geomspace", "1", "9", "4"])

    job = {
        "job_id": str(uuid4()),
        "type": "script",
        "is_calibration_sup_job": True,
        "name": "demodulation_scenario",
        "params": {
            "Sine - Frequency": signal_array,
            "Demod - Modulation frequency": demod_array,
        },
    }
    return job


def gen_array(option):

    fn_dispatcher = {
        "linspace": np.linspace,
        "geomspace": np.geomspace,
        "logspace": np.logspace,
    }

    if option[0] == "stepspace":
        return ("start=" + option[1], "stop=" + option[2], "step=" + option[3])
    else:
        numpy_array = fn_dispatcher[option[0]](
            *(list(map(float, option[1:3])) + [int(option[3])])
        )
        return numpy_array.tolist()
