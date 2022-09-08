# This code is part of Tergite
#
# (C) Copyright Abdullah-Al Amin 2021
# (C) Copyright David Wahlstedt 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import json
import pathlib
import settings
from tempfile import gettempdir
from uuid import uuid4

import numpy as np
import requests

import measurement_jobs.measurement_jobs as measurement_jobs

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
    job = measurement_jobs.mk_job_pulsed_res_spect(
        # Mandatory parameters
        num_pts=201,
        qa_avg=1024,
        readout_amp=25e-3,
        readout_freq_start=5.99e9,
        readout_freq_stop=6.01e9,
        readout_power=10,
        # optional argument for calibration supervisor
        is_calibration_sup_job=False,  # default True
        # non-mandatory arguments overriding defaults
        # just to show we can override: default in Toml file is 274.68e6
        drive_freq=274.68e6 + 0.001e6,
    )
    return job

# Jobs can be generated directly as follows, and then default
# parameters can be overridden. However, with this method there is no
# control that mandatory arguments are provided.
def generate_job_direct():
    job = {
        "job_id": str(uuid4()),
        "type": "script",
        "name": "pulsed_resonator_spectroscopy",
        # Defaults for "params" are loaded in scenario_scripts.py from
        # measurement_jobs/parameter_defaults/vna_resonator_spectroscopy.toml
        "params": {},
    }
    return job


"""
    Parameters (needs to be updated)
    ---------
          ##  _range_type: sweeping direction, valid values: "Start - Stop", "Span", "Single"

            "qa_avg": number of recording average
            "num_pts": number of points of measurement
            "readout_amp": Readout MQPG drive signal amplitude, used when "Single" or "Span" range type is used

            "start_freq: Start of sweeping frequency
            "stop_freq": End of sweeping frequency

            "power": used when "Single" or "Span" range type is used
        """


if __name__ == "__main__":
    main()
