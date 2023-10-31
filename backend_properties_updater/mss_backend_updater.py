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

import json
import toml
import settings
import requests
from backend_properties_storage.storage import get_component_value

mss_url = str(settings.MSS_MACHINE_ROOT_URL)
backend_settings = settings.BACKEND_SETTINGS


# ==================================================
# Creates a dictionary resambling the mongodb
# collection of backend, this backend is updated
# by mss thruogh put("/backends") rest-api
# endpoint directly with dictionary converted json.
# ==================================================


def create_backend_snapshot() -> dict:
    with open(backend_settings, "r") as f:
        config = toml.load(f)
        general_config = config["general_config"]
        qubit_ids = config["device_config"]["qubit_ids"]
        qubit_parameters = config["device_config"]["qubit_parameters"]
        resonator_parameters = config["device_config"]["resonator_parameters"]
        coupling_map = config["device_config"]["coupling_map"]
        gate_configs = config["gates"]

    # updating and constructing components
    qubits = []
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

    resonators = []
    for qubit_id in qubit_ids:
        id = str(qubit_id).strip("q")
        resonator = {}
        for parameter in resonator_parameters:
            if parameter == "id":
                value = qubit_id
            else:
                # reading the component parameter values in redis
                value = get_component_value("readout_resonator", parameter, id)
            resonator.update({parameter: value})
        resonators.append(resonator)

    # more componets, like couplers etc. can be added in similar manner and added
    # to the device_properties dict ....

    device_properties = {
        "device_properties": {**{"qubit": qubits}, **{"readout_resonator": resonators}}
    }
    return {
        **general_config,
        **{"qubit_ids": qubit_ids},
        **device_properties,
        **{"coupling_map": coupling_map},
        **{"gates": gate_configs},
    }


def update_mss(collection:str):
    current_backend_snapshot = create_backend_snapshot()
    backend_snapshot_json = json.dumps(current_backend_snapshot, indent=4)
    response = requests.put(mss_url + "/backends" + f"/{collection}", backend_snapshot_json)
    if response:
        print(f"'{current_backend_snapshot['name']}' backend configuration is sent to mss")
    else:
        print(f"Could not send '{current_backend_snapshot['name']} 'backend configuration to mss")
