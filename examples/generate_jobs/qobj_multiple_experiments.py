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


import json
import pathlib
from uuid import uuid4

import requests

import settings

# settings
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL

REST_API_MAP = {"jobs": "/jobs"}


def main():
    job = {
        "job_id": str(uuid4()),
        "type": "script",
        "name": "qiskit_qasm_runner",
        "post_processing": "process_qiskit_qasm_runner_qasm_dummy_job",
        "params": {
            "qobj": {
                "qobj_id": "exp123_072018",
                "schema_version": "1.0.0",
                "type": "QASM",
                "header": {
                    "description": "Set of Experiments 1",
                    "backend_name": "pingu",
                },
                "config": {"shots": 1024, "memory_slots": 1, "init_qubits": True},
                "experiments": [
                    {
                        "header": {
                            "memory_slots": 1,
                            "n_qubits": 3,
                        },
                        "config": {},
                        "instructions": [
                            {"name": "ry", "qubits": [1], "params": [0.8]},
                            {"name": "h", "qubits": [0], "params": []},
                        ],
                    },
                    {
                        "header": {
                            "memory_slots": 1,
                            "n_qubits": 3,
                        },
                        "config": {},
                        "instructions": [
                            {"name": "rx", "qubits": [0], "params": [0.8]},
                            {"name": "y", "qubits": [1], "params": []},
                            {"name": "h", "qubits": [1], "params": []},
                        ],
                    },
                ],
            },
        },
        "hdf5_log_extraction": {"waveforms": True, "voltages": True},
    }

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


if __name__ == "__main__":
    main()
