# This code is part of Tergite
#
# (C) Copyright Abdullah-Al Amin 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Modified:
#
# - Martin Ahindura, 2023
# - Stefan Hill, 2024
#
# CAUTION: This updater is currently also used in the tergite-autocalibration-lite repository!
# Any change on this file should be done in both repositories until they are eventually merged!

import json
from os import path
from pathlib import Path
from typing import Mapping, Any

import requests
import toml

import settings
from app.utils.storage import get_component_value

mss_url = str(settings.MSS_MACHINE_ROOT_URL)
_project_root = path.dirname(
    path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
)
backend_settings = Path(_project_root) / settings.BACKEND_SETTINGS


def create_backend_snapshot() -> dict:
    """Creates a dict containing the properties of this backend

    This dictionary is later saved in the MSS storage by
    PUTing it to the `/backends` MSS endpoint
    """
    with open(backend_settings, "r") as f:
        config = toml.load(f)
        general_config = config["general_config"]
        qubit_ids = config["device_config"]["qubit_ids"]
        qubit_parameters = config["device_config"]["qubit_parameters"]
        resonator_parameters = config["device_config"]["resonator_parameters"]
        discriminator_parameters = config["device_config"]["discriminator_parameters"][
            "lda_parameters"
        ]
        coupling_map = config["device_config"]["coupling_map"]
        meas_map = config["device_config"]["meas_map"]
        gate_configs = config["gates"]

    # updating and constructing components
    qubits = []
    resonators = []
    lda_discriminators = {}

    for qubit_id in qubit_ids:
        id = str(qubit_id).strip("q")
        qubit = {}
        for parameter in qubit_parameters:
            if parameter == "id":
                value = qubit_id
            else:
                # reading the component parameter values in redis
                value = get_component_value("qubit", parameter, id)
            qubit.update({parameter: value})
        qubits.append(qubit)

        resonator = {}
        for parameter in resonator_parameters:
            if parameter == "id":
                value = qubit_id
            else:
                # reading the component parameter values in redis
                value = get_component_value("readout_resonator", parameter, id)
            resonator.update({parameter: value})
        resonators.append(resonator)

        # Here, we are doing it only for lda
        lda_discriminators[qubit_id] = {
            parameter: get_component_value("discriminator", parameter, id)
            for parameter in discriminator_parameters
        }

    # more components, like couplers etc. can be added in similar manner and added
    # to the device_properties dict ....

    device_properties = {
        "device_properties": {**{"qubit": qubits}, **{"readout_resonator": resonators}}
    }
    return {
        **general_config,
        **{"qubit_ids": qubit_ids},
        **device_properties,
        **{"discriminators": {"lda": lda_discriminators}},
        **{"coupling_map": coupling_map},
        **{"meas_map": meas_map},
        **{"gates": gate_configs},
    }


def post_mss_backend(backend_json: Mapping[str, Any] = None, collection: str = None):
    """
    Push a backend definition to the MSS endpoint

    Args:
        backend_json: Backend definition as JSON object
                      Will be automatically generated if not provided
        collection: Please specify if backend should not be pushed to the standard collection in the DB

    Returns:

    """
    if backend_json is None:
        backend_json = create_backend_snapshot()
    backend_snapshot_json = json.dumps(backend_json, indent=4)
    if collection:
        response = requests.put(
            mss_url + f"/backends?collection={collection}", backend_snapshot_json
        )
    else:
        response = requests.put(mss_url + "/backends", backend_snapshot_json)

    if response:
        print(f"'{backend_json['name']}' backend configuration is sent to mss")
    else:
        print(f"Could not send '{backend_json['name']} 'backend configuration to mss")
