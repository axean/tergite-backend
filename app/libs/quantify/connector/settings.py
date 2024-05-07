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
import enum
from pathlib import Path
from typing import List, Optional, Set, Dict, Any, Tuple

import quantify_core.data.handling as dh
from pydantic import BaseModel, Extra, validator
from pydantic.fields import ModelField
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


class QuantifyConfigItem(BaseModel, extra=Extra.allow):
    """configuration item in quantify config"""

    def to_quantify(
        self, exclude_none: bool = True, exclude: Optional[Set] = None
    ) -> Dict[str, Any]:
        """Converts to quantify configuration dict

        Args:
            whether None should be excluded from the configuration

        Returns:
            dict that would be part of the quantify configuration
        """
        raw_dict = self.dict(exclude_none=exclude_none, exclude=exclude)
        return {
            k: v.to_quantify() if isinstance(v, QuantifyConfigItem) else v
            for k, v in raw_dict.items()
        }


class PortClockConfig(QuantifyConfigItem):
    """configurations for the port and clock for an input/output"""

    port: Optional[str] = None
    clock: Optional[str] = None
    mixer_amp_ratio: Optional[float] = None
    mixer_phase_error_deg: Optional[float] = None
    interm_freq: Optional[float] = None
    ttl_acq_threshold: Optional[float] = None
    init_offset_awg_path_I: Optional[float] = None
    init_offset_awg_path_Q: Optional[float] = None
    init_gain_awg_path_I: Optional[float] = None
    init_gain_awg_path_Q: Optional[float] = None
    qasm_hook_func: Optional[float] = None


class Channel(QuantifyConfigItem):
    """Input/output channels for cluster modules"""

    marker_debug_mode_enable: Optional[bool] = None
    portclock_configs: List[PortClockConfig] = []
    dc_mixer_offset_I: Optional[float] = None
    dc_mixer_offset_Q: Optional[float] = None
    input_gain_I: Optional[float] = None
    input_gain_Q: Optional[float] = None
    input_gain_0: Optional[float] = None
    input_gain_1: Optional[float] = None
    input_att: Optional[float] = None
    output_att: Optional[float] = None
    lo_name: Optional[str] = None
    lo_freq: Optional[float] = None
    downconverter_freq: Optional[float] = None
    mix_lo: Optional[bool] = True


class _ClusterRef(str, enum.Enum):
    """Reference sources for clusters; typically 10 MHz clocks"""

    INTERNAL = "internal"
    EXTERNAL = "external"


class _ClusterModuleType(str, enum.Enum):
    """Types of cluster modules"""

    QCM = "QCM"
    QRM = "QRM"
    QCM_RF = "QCM_RF"
    QRM_RF = "QRM_RF"


_MODULE_TYPE_VALID_CHANNELS_MAP: Dict[_ClusterModuleType, Dict[str, bool]] = {
    _ClusterModuleType.QCM: {
        "complex_outputs": True,
        "real_outputs": True,
        "digital_inputs": True,
    },
    _ClusterModuleType.QRM: {
        "complex_outputs": True,
        "complex_inputs": True,
        "real_outputs": True,
        "real_inputs": True,
        "digital_inputs": True,
    },
    _ClusterModuleType.QCM_RF: {
        "complex_outputs": True,
        "digital_inputs": True,
    },
    _ClusterModuleType.QRM_RF: {
        "complex_outputs": True,
        "complex_inputs": True,
        "digital_inputs": True,
    },
}


class ClusterModule(QuantifyConfigItem):
    """General configration for a cluster module"""

    name: str
    instrument_type: _ClusterModuleType
    complex_outputs: Optional[List[Channel]] = None
    complex_inputs: Optional[List[Channel]] = None
    real_outputs: Optional[List[Channel]] = None
    real_inputs: Optional[List[Channel]] = None
    digital_outputs: Optional[List[Channel]] = None

    _channel_list_fields: Tuple[str] = (
        "complex_outputs",
        "complex_inputs",
        "real_outputs",
        "real_inputs",
        "digital_inputs",
    )

    @validator(
        "complex_outputs",
        "complex_inputs",
        "real_outputs",
        "real_inputs",
        "digital_inputs",
    )
    def valid_channels_for_instrument_type(
        cls, v, values, config, field: ModelField, **kwargs
    ):
        try:
            module_type = values["instrument_type"]
        except KeyError:
            return v

        valid_channels = _MODULE_TYPE_VALID_CHANNELS_MAP[module_type]

        if not valid_channels.get(field.name):
            raise ValueError(
                f"'{field.name}' are not permitted in cluster modules of type '{module_type}'"
            )

        return v

    def to_quantify(
        self, exclude_none: bool = True, exclude: Optional[Set] = None
    ) -> Dict[str, Any]:
        """Converts this cluster module into a quantify config

        It returns something like, where n is zero-based index i.e. starting from 0:
            {
                instrument_type: "...",
                <channel_field_name>_<n>: {
                    portclock_configs: [
                        {...},
                        {...}
                    ]
                },
                 <channel_field_name>_<n>: {
                    portclock_configs: [
                        {...},
                        {...}
                    ]
                } #...
            }
        """
        excluded_fields = exclude if isinstance(exclude, set) else set()

        # exclude some fields that are only in place
        # to make it easier to configure the hardware
        excluded_fields.update(set(self._channel_list_fields))

        # exclude some fields that are just for show
        excluded_fields.update({"name"})

        raw_dict = super().to_quantify(
            exclude_none=exclude_none, exclude=excluded_fields
        )

        # add the fields that are required for the hardware config
        for channel_list_field in self._channel_list_fields:
            # change the plural form from the singular form
            channel_field = channel_list_field.rstrip("s")
            channel_confs = getattr(self, channel_list_field, __default=[])
            raw_dict.update(
                {
                    f"{channel_field}_{index}": value.to_quantify()
                    for index, value in enumerate(channel_confs)
                }
            )

        return raw_dict


class Cluster(QuantifyConfigItem):
    """Configuration for the cluster"""

    name: str
    ref: _ClusterRef = _ClusterRef.EXTERNAL

    instrument_type: str = "Cluster"
    sequence_to_file: bool = False
    modules: List[ClusterModule] = []

    def to_quantify(
        self, exclude_none: bool = True, exclude: Optional[Set] = None
    ) -> Dict[str, Any]:
        """Converts this cluster into a quantify config

        It returns something like, where n is one-based index i.e. starting from 1:
        {
            instrument_type: "Cluster",
            ref: "external",
            sequence_to_file: False,
            <cluster name>_module<n>: {...},
            <cluster name>_module<n>: {...},
        }
        """
        excluded_fields = exclude if isinstance(exclude, set) else set()

        # exclude some fields that are only in place
        # to make it easier to configure the hardware
        excluded_fields.update({"name", "modules"})

        raw_dict = super().to_quantify(
            exclude_none=exclude_none, exclude=excluded_fields
        )

        # add the fields that are required for the hardware config
        cluster_name = self.name
        raw_dict.update(
            {
                f"{cluster_name}_module{index + 1}": value.to_quantify()
                for index, value in enumerate(self.modules)
            }
        )

        return raw_dict


class LocalOscillator(QuantifyConfigItem):
    """Configuration for the local oscillator"""

    name: str
    instrument_type: str = "LocalOscillator"
    frequency: Optional[float] = None
    power: float

    def to_quantify(
        self, exclude_none: bool = False, exclude: Optional[Set] = None
    ) -> Dict[str, Any]:
        """Converts this cluster into a quantify config

        It returns something like:
        {
            instrument_type: "LocalOscillator",
            frequency: 5e9,
            power: 20,
        }
        """
        excluded_fields = exclude if isinstance(exclude, set) else set()

        # exclude name as it is not part of the quantify configuration schema
        excluded_fields.update({"name"})
        return self.dict(exclude=excluded_fields, exclude_none=exclude_none)
