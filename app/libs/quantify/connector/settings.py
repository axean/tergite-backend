# This code is part of Tergite
#
# (C) Axel Andersson (2022)
# (C) Martin Ahindura (2024)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from pathlib import Path

import quantify_core.data.handling as dh
from starlette.config import Config

from ..utils.general import load_config


class ConnectorSettings:
    def __init__(self: "ConnectorSettings", /, *, env_file: Path):
        # NOTE: shell env variables take precedence over the configuration file
        config = Config(env_file)
        DATA_DIR = config("DATA_DIR", cast=Path)

        # Tell Quantify where to store data
        dh.set_datadir(DATA_DIR)

        # List of names of QCoDeS devices to connect to
        self.DEVICES = list(
            map(lambda n: str.strip(n), str.split(config("DEVICES", cast=str), ","))
        )

        # Construct hardware configuration file from device settings files
        HARDWARE_CONFIG = [
            config(f"{name}_SETTINGS", cast=load_config) for name in self.DEVICES
        ]
        tmp = dict()
        for dev_cfg in HARDWARE_CONFIG:
            tmp = {**tmp, **dev_cfg}
        self.HARDWARE_CONFIG = tmp
        self.HARDWARE_CONFIG = {
            **self.HARDWARE_CONFIG,
            "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        }
        self.SIMULATE = config("SIMULATE", cast=bool, default=False)
        if config("DUMMY_CFG", default=False):
            self.DUMMY_CFG = list(
                map(
                    lambda n: str.strip(n),
                    str.split(config("DUMMY_CFG", cast=str), ","),
                )
            )

        self.simulator_type = config("SIMULATOR_TYPE", default="scqt").lower()
