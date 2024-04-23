# This code is part of Tergite
#
# (C) Copyright Simon Genne, Arvid Holmqvist, Bashar Oumari, Jakob Ristner,
#               Björn Rosengren, and Jakob Wik 2022 (BSc project)
# (C) Copyright Fabian Forslund, Nicklas Botö 2022
# (C) Copyright Abdullah-Al Amin 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import json
from time import sleep

import redis

from archive.backend_properties_updater.backend_config_updater import config_updater

with open("backend_config.json") as config_json:
    config = json.load(config_json)
    qubit_configs = config["qubits"]
    resonator_configs = config["resonators"]
    coupling_map = config["coupling_map"]
    gate_configs = config["gates"]
    config_snapshot_info = {
        key: config[key]
        for key in [
            "backend_name",
            "backend_version",
            "online_date",
            "description",
            "sample_name",
            "n_qubits",
        ]
    }
    generator = config_updater(
        qubit_configs, resonator_configs, gate_configs, coupling_map
    )
    r = redis.Redis()
    r.set("config", json.dumps(config))


def main():
    while True:
        generator.config_update()
        generated_data = generator.to_dict()
        current_snapshot = {**config_snapshot_info, **generated_data}
        r.set("current_snapshot", json.dumps(current_snapshot))
        print("backend configuration updated by randomly generated values")
        sleep(300)


if __name__ == "__main__":
    main()
