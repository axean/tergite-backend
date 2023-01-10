# This code is part of Tergite
#
# (C) Copyright Andreas Bengtsson, Miroslav Dobsicek 2020
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
from tempfile import gettempdir
from uuid import uuid4

import requests

import measurement_jobs.measurement_jobs as measurement_jobs
import settings

# Settings
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

    job = measurement_jobs.mk_job_res_spect_vna(
        freq_start=6.0e9,
        freq_stop=7.0e9,
        num_pts=10001,
        # For multiple power sweeps use: [start, stop, n_pts],
        # example:[-50, 0, 1]
        power=[-50, 0, 51],
        # No post-processing will take place. This measurement does not
        # exactly the same as what will be phase one or two in VNA
        # resonator spectroscopy, and there is yet no decision made
        # how to post-process this one.
        post_processing=None,
        # optional argument for calibration supervisor
        is_calibration_supervisor_job=False,  # default True
        # non-mandatory arguments overriding defaults
    )
    return job


# Jobs can be generated directly as follows, and then default
# parameters can be overridden. However, with this method there is no
# control that mandatory arguments are provided.
def generate_job_direct():

    job = {
        "job_id": str(uuid4()),
        "type": "script",
        "name": "resonator_spectroscopy",
        # Defaults for "params" are loaded in scenario_scripts.py from
        # measurement_jobs/parameter_defaults/vna_resonator_spectroscopy.toml
        "params": {
            "freq_start": 6.0e9 + 0.01e9,  # demonstrating it can be overridden
        },
    }
    return job


"""
    Parameters
    ----------
    freq_start : (float) start sweep frequency [Hz]
    freq_stop  : (float) stop sweep frequency [Hz]
    if_bw   : (float) IF bandwidth setting [Hz]
    num_pts : (int) number of frequency points
    power   : [float, float, int] output power of VNA [dBm, dBm, n_pts]
      for single power measurement use single element, e.g.: [0],
      for multiple power sweep use: [p_start, p_stop, n_pts], e.g.:[-50, 0, 51]
    num_ave : (int) number of averages
"""


if __name__ == "__main__":
    main()
