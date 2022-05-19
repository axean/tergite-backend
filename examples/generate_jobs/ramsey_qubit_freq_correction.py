# This code is part of Tergite
#
# (C) Copyright Abdullah-Al Amin 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import numpy as np
import json
from uuid import uuid4
import requests
import pathlib
from tempfile import gettempdir
import settings

# INSTRUMENT = ZI

# settings
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL

REST_API_MAP = {"jobs": "/jobs"}


def main():
    job = generate_job()

    temp_dir = gettempdir()
    file = pathlib.Path(temp_dir) / str(uuid4())
    with file.open("w") as dest:
        json.dump(job, dest)

    with file.open("r") as src:
        files = {"upload_file": src}
        url = str(BCC_MACHINE_ROOT_URL) + REST_API_MAP["jobs"]
        response = requests.post(url, files=files)

        if response:
            print("Job has been successfully sent")

    file.unlink()


def generate_job():

    job = {
        "job_id": str(uuid4()),
        "type": "script",
        "name": "ramsey_qubit_freq_correction",
        "params": {
            # All params are general form of Pulsed Resonator Spectroscopy,
            # Two-tone, Rabi and Ramsey
            # The important parameters for current job type are markd with # &
            "control_output_enabled": 1,  # & output On, exiting the qubit
            "control_freq_range_type": "Single",
            "control_start_freq": 270e6,
            "control_stop_freq": 280e6,
            "control_if_freq": 4.5e9,  # & should be in some config param
            "control_freq": 273.18e6,  # & qubiy exitation frequency = "control_freq" + "control_if_freq"
            # should be replaced by the measured variable
            "control_amp_range_type": "Single",
            "control_amp": 48e-3,  # & Half the pi-pulse amplitude, should fetched from redis
            "control_amp_start": 1e-3,
            "control_amp_stop": 200e-3,
            "control_duration": 100e-9,
            "control_pulse_spacing_range_type": "Start - Stop",  # & doing a pulse spacing sweep
            "control_pulse_spacing_start": 100e-9,  # &
            "control_pulse_spacing_stop": 2e-6,  # &
            "control_pulse_spacing": 2e-6,  # kept as default value
            "control_power_range_type": "Single",
            "control_power": -10,  # & used in RS-RF Source
            "control_start_power": -20,
            "control_stop_power": 0,
            "readout_freq": 379.17e6,  # & close to expected resonance frequency
            "readout_freq_range_type": "Single",  # & single fequency at the resosnace of the resonator
            "readout_start_freq": 5.99e9,
            "readout_stop_freq": 6.01,
            "readout_resonance_freq": 5.9999e9,  # & resonace determined by pulsed_res_spect
            "readout_amp_range_type": "Single",
            "readout_amp": 25e-3,  # &
            "readout_amp_start": 5e-3,
            "readout_amp_stop": 20e-3,
            "readout_duration": 3e-6,
            "readout_sampling_delay": 200e-9,
            "readout_power_range_type": "Single",
            "readout_power": 16,  # & used in RS-RF Source
            "readout_start_power": -20,
            "readout_stop_power": 0,
            "num_pts": 191,  # & 100 ns to 2 us, 191 points = 10 ns sweep resolation
            "num_pts_other_axis": 21,  # used when another measurement sweep is required
            "sampling_duration": 2e-6,
            "repetition_delay": 1e-6,
            "qa_integration_length": 4096,
            "qa_delay": 1e3,
            "qa_avg": 1024,
            "integration_window_start": 1e-6,
            "integration_window_stop": 3e-6,
            "trace_time": 5e-6,
            "mqpg_smpl_rate": 2.4e9,
            "hdawg_trigger_range_type": "Single",
            "hdawg_trigger_period_start": 300e-6,
            "hdawg_trigger_period_stop": 600e-6,
            "hdawg_int_trig_period": 300e-6,  # aka relaxation_time
            "hdawg_marker_start_time": 4.9e-6,
            "hdwag_marker_duration": 100e-9,
        },
    }
    return job


"""
    Parameters
    ---------
          ##  _range_type: sweeping direction, valid values: "Start - Stop", "Span", "Single"
            
            "freq_range_type": Frequency sweeping type
            "start_freq: Start of sweeping frequency
            "stop_freq": End of sweeping frequency
            "center_freq": Center of sweeping frequency
            "span_freq": Frequency Span value from center_freq

            "power_range_type": Power sweeping type
            "power": used when "Single" or "Span" range type is used
            "start_power": Start of sweeping
            "stop_power": End of sweeping

            "control_output_enabled": RF-Source output used to drive qubit. 0=Off, 1=On 
            "control_amp_range_type": Control MQPG drive signal amplitude sweeping type
            "control_amp": Control MQPG drive signal amplitude, used when "Single" or "Span" range type is used
            "control_amp_start": Start of amplitude sweeping
            "control_amp_stop": Stop of amplitude sweeping
            "control_duration": Control MQPG drive pulse duration 
            "control_if_freq": Control RF-Source frequency
            
            "num_pts": number of points of measurement

            "readout_freq": Readout MQPG frequency
            "readout_amp_range_type": Readout MQPG drive signal amplitude sweeping type
            "readout_amp": Readout MQPG drive signal amplitude, used when "Single" or "Span" range type is used
            "readout_duration": Readout MQPG output pulse duration
            "readout_sampling_delay": A delay that depends on the readout path length

            "sampling_duration": signal recording duration
            "repetition_delay": 
            "qa_integration_length": intregation of samples to have energy value
            "qa_delay": delay just after recording of pulse, to avoid noise in pulse start point
            "qa_avg": number of recording average 
            "integration_window_start": Defines strat time when a window is used to define integration scope
            "integration_window_stop": Defines stop time .. as above
            
            "trace_time": MPQG constructed signal length in time
            "mqpg_smpl_rate": DAC sample rate

            "hdawg_int_trig_period": minimum time to start another control pulse # aka relaxation_time
            "hdawg_marker_start_time": defines start time of Marker output to signal Readout process 
            "hdwag_marker_duration": Marker pulse duration


        """


if __name__ == "__main__":
    main()
