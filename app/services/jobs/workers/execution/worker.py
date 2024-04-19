# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020, 2021
# (C) Copyright Abdullah-Al Amin 2021
# (C) Copyright Axel Andersson 2022
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
import logging
from pathlib import Path
from uuid import uuid4

import requests

import settings

from ...service import Location, inform_failure, inform_location

# Settings
STORAGE_ROOT = settings.STORAGE_ROOT
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL
QUANTIFY_MACHINE_ROOT_URL = settings.QUANTIFY_MACHINE_ROOT_URL

REST_API_MAP = {"scenarios": "/scenarios", "qobj": "/qobj"}


def post_schedule_file(job_dict: dict, /):
    print(f"Received OpenPulse schedule")

    tmp_file = Path(STORAGE_ROOT) / (str(uuid4()) + ".to_quantify")

    with tmp_file.open("w") as store:
        json.dump(job_dict, store)  # copy incoming data to temporary file

    with tmp_file.open("r") as source:
        files = {
            "upload_file": (tmp_file.name, source),
            "send_logfile_to": (None, str(BCC_MACHINE_ROOT_URL)),
        }

        url = str(QUANTIFY_MACHINE_ROOT_URL) + REST_API_MAP["qobj"]
        print("Sending the pulse schedule to Quantify")
        try:
            response = requests.post(url, files=files)
        except Exception as exp:
            logging.error(exp)
            response = requests.Response()
            response.status_code = 500

    tmp_file.unlink()
    return response


def job_execute(job_file: Path):
    print(f"Executing file {str(job_file)}")

    with job_file.open() as f:
        job_dict = json.load(f)

    job_id = job_dict["job_id"]

    # Inform supervisor
    inform_location(job_id, Location.EXEC_W)

    response = post_schedule_file(job_dict)

    if response.ok:
        # clean up
        job_file.unlink(missing_ok=True)

        print("Job executed successfully")
        return {"message": "ok"}

    # failure case: response received but it carries error code (4xx or 5xx)
    elif not response.ok:
        print("Job failed")
        print(f"Server rejected job. Response: {response}")
        inform_failure(job_id, reason=f"HTTP error code: {response.status_code}")
        return {"message": "failed"}

    # failure case: Unknown script name
    elif response is None:
        print("Job failed")
        print(f"Unknown script name {job_dict['name']}")
        inform_failure(job_id, reason="unknown script name")
        return {"message": "failed"}

    # failure case: Unspecified error
    else:
        print("Job failed")
        # inform supervisor about failure
        inform_failure(job_id, reason="no response")
        return {"message": "failed"}
