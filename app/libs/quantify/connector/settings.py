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
import os
from pathlib import Path
from typing import Union, List, Dict, Any

import quantify_core.data.handling as dh
from pydantic import BaseModel
from starlette.config import Config

from ..utils.config import QuantifyConfig, SimulatorType
from ..utils.general import load_config


class ConnectorSettings(BaseModel):
    DEVICES: List[str] = []
    DUMMY_CFG: List[str] = []
    HARDWARE_CONFIG: Dict[str, Any] = {}
    SIMULATE: bool = False
    simulator_type: str = f"{SimulatorType.SCQT}"

    @classmethod
    def from_env(cls, env_file: Path) -> "ConnectorSettings":
        """The legacy way of loading the configuration from the environment file"""
        # NOTE: shell env variables take precedence over the configuration file
        config = Config(env_file)
        DATA_DIR = config("DATA_DIR", cast=Path)

        # Tell Quantify where to store data
        dh.set_datadir(DATA_DIR)

        # List of names of QCoDeS devices to connect to
        devices = list(
            map(lambda n: str.strip(n), str.split(config("DEVICES", cast=str), ","))
        )

        # Construct hardware configuration file from device settings files
        hardware_config = [
            config(f"{name}_SETTINGS", cast=load_config) for name in devices
        ]
        tmp = dict()
        for dev_cfg in hardware_config:
            tmp = {**tmp, **dev_cfg}
        hardware_config = {
            **tmp,
            "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        }
        simulate = config("SIMULATE", cast=bool, default=False)
        dummy_cfg = []

        if config("DUMMY_CFG", default=False):
            dummy_cfg = list(
                map(
                    lambda n: str.strip(n),
                    str.split(config("DUMMY_CFG", cast=str), ","),
                )
            )

        simulator_type = config("SIMULATOR_TYPE", default="scqt").lower()

        return cls(
            DEVICES=devices,
            DUMMY_CFG=dummy_cfg,
            HARDWARE_CONFIG=hardware_config,
            SIMULATE=simulate,
            simulator_type=simulator_type
        )

    @classmethod
    def load_yaml_config(cls, config_file: Union[str, bytes, os.PathLike]) -> "ConnectorSettings":
        """Loads the config from the YAML config file"""
        conf = QuantifyConfig.from_yaml(config_file)

        # Tell Quantify where to store data
        dh.set_datadir(conf.general.data_directory)

        return cls(
            HARDWARE_CONFIG=conf.to_quantify(),
            SIMULATE=conf.general.is_simulator,
            simulator_type=f"{conf.general.simulator_type}".lower()
        )
