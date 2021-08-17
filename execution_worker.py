# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020, 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from pathlib import Path
import json
import time
from scenario_scripts import demodulation_scenario, qobj_scenario, qobj_dummy_scenario
import requests
import settings

# settings
STORAGE_ROOT = settings.STORAGE_ROOT
LABBER_MACHINE_ROOT_URL = settings.LABBER_MACHINE_ROOT_URL
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL


REST_API_MAP = {"scenarios": "/scenarios"}


def job_execute(job_file: Path):
    print(f"Executing file {str(job_file)}")

    # extract job_id from the filename
    job_id = job_file.stem
    scenario_file = Path(STORAGE_ROOT) / (job_id + ".labber")

    job_dict = {}
    with job_file.open() as f:
        job_dict = json.load(f)

    print(f"Job script type: {job_dict['name']}")
    if job_dict["name"] == "demodulation_scenario":
        signal_array = job_dict["params"]["Sine - Frequency"]
        demod_array = job_dict["params"]["Demod - Modulation frequency"]

        scenario = demodulation_scenario(signal_array, demod_array)

        scenario.log_name = "Test signal demodulation - " + job_id
        # scenario.save("/tmp/my.json", save_as_json=True)

    elif job_dict["name"] == "qiskit_qasm_runner":
        scenario = qobj_scenario(job_dict)

        scenario.log_name += job_id
    elif job_dict["name"] == "qasm_dummy_job":
        scenario = qobj_dummy_scenario(job_dict)

        scenario.log_name += job_id

    else:
        print(f"Unknown script name {job_dict['name']}")
        print("Job failed")
        return {"message": "failed"}

    # Store important information inside the scenario: using the tag list
    # 1) job_id
    # 2) script name
    scenario.tags.tags = [job_id, job_dict["name"]]

    scenario.save(scenario_file)
    print(f"Scenario generated at {str(scenario_file)}")

    with scenario_file.open("rb") as source:
        files = {
            "upload_file": (scenario_file.name, source),
            "send_logfile_to": (None, str(BCC_MACHINE_ROOT_URL)),
        }
        url = str(LABBER_MACHINE_ROOT_URL) + REST_API_MAP["scenarios"]
        print("Sending the scenario to Labber")
        response = requests.post(url, files=files)

    if response:
        # clean up
        job_file.unlink()
        scenario_file.unlink()

        print("Scenario job executed successfully")
        return {"message": "ok"}
    else:
        print("Failed")
        return {"message": "failed"}
