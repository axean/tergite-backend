# This code is part of Tergite
#
# (C) Copyright Andreas Bengtsson, Miroslav Dobsicek 2020
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
import settings

# settings
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL

REST_API_MAP = {"jobs": "/jobs"}


def main():
    job = generate_job()

    file = pathlib.Path("/tmp") / str(uuid4())
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
    qobj = generate_qobj()
    job = {
        "job_id": str(uuid4()),
        "type": "script",
        "name": "qasm_dummy_job",
        "params": {"qobj": qobj},
    }
    return job

def generate_qobj():

    qobj = {
        "config": {
            "max_credits": 10,
            "memory": False,
            "memory_slots": 4,
            "n_qubits": 4,
            "parameter_binds": [],
            "parametric_pulses": [],
            "shots": 1024,
        },
        "experiments": [
            {
                "config": {"memory_slots": 3, "n_qubits": 3},
                "header": {
                    "clbit_labels": [["c", 0], ["c", 1], ["c", 2]],
                    "creg_sizes": [["c", 3]],
                    "global_phase": 0,
                    "memory_slots": 3,
                    "n_qubits": 3,
                    "name": "circuit0",
                    "qreg_sizes": [["q", 3]],
                    "qubit_labels": [["q", 0], ["q", 1], ["q", 2]],
                },
                "instructions": [
                    {"name": "u2", "params": [0.0, 3.141592653589793], "qubits": [1]},
                    {
                        "name": "u3",
                        "params": [
                            3.141592653589793,
                            3.141592653589793,
                            3.141592653589793,
                        ],
                        "qubits": [2],
                    },
                    {"memory": [0], "name": "measure", "qubits": [0]},
                    {"memory": [1], "name": "measure", "qubits": [1]},
                    {"memory": [2], "name": "measure", "qubits": [2]},
                ],
            }
        ],
        "header": {"backend_name": "qasm_simulator", "backend_version": "2.0.0"},
        "qobj_id": "690fe004-cde4-40b6-9ff0-d389480c3c7e",
        "schema_version": "1.3.0",
        "type": "QASM",
    }
    return qobj


if __name__ == "__main__":
    main()
