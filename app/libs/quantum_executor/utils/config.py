# This code is part of Tergite
#
# (C) Martin Ahindura (2024)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#

"""Utilities concerned with the configuration of the Quantum hardware/software"""
import enum
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Extra, validator
from pydantic.fields import ModelField
from quantify_scheduler.backends.qblox import instrument_compilers as qblox_compiler
from ruamel.yaml import YAML

from app.libs.quantum_executor.utils.general import get_duplicates

yaml = YAML(typ="safe")


class ExecutorConfigItem(BaseModel, extra=Extra.allow):
    """configuration item in executor config"""

    def to_quantify(
        self, exclude_none: bool = True, exclude: Optional[Set] = None
    ) -> Dict[str, Any]:
        """Converts to quantify-scheduler-compatible configuration dict

        https://quantify-os.org/docs/quantify-scheduler/dev/reference/qblox/Cluster.html

        Args:
            whether None should be excluded from the configuration

        Returns:
            dict that would be part of the quantify configuration
        """
        raw_dict = self.dict(exclude_none=exclude_none, exclude=exclude)
        return {
            k: v.to_quantify() if isinstance(v, ExecutorConfigItem) else v
            for k, v in raw_dict.items()
        }


class PortClockConfig(ExecutorConfigItem):
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


class Channel(ExecutorConfigItem):
    """Input/output channels for cluster modules"""

    name: str
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
    mix_lo: Optional[bool] = None

    def to_quantify(
        self, exclude_none: bool = True, exclude: Optional[Set] = None
    ) -> Dict[str, Any]:
        """Converts to quantify-scheduler-compatible configuration dict

        Args:
            whether None should be excluded from the configuration

        Returns:
            dict that would be part of the quantify configuration
        """
        excluded_fields = exclude if isinstance(exclude, set) else set()
        # exclude some fields that are only in place
        # to make it easier to configure the hardware
        excluded_fields.update({"name"})

        return super().to_quantify(exclude_none=exclude_none, exclude=excluded_fields)


class _ClusterRef(str, enum.Enum):
    """Reference sources for clusters; typically 10 MHz clocks"""

    INTERNAL = "internal"
    EXTERNAL = "external"


class ClusterModuleType(str, enum.Enum):
    """Types of cluster modules"""

    QCM = "QCM"
    QRM = "QRM"
    QCM_RF = "QCM_RF"
    QRM_RF = "QRM_RF"


_MODULE_TYPE_VALID_CHANNELS_MAP: Dict[ClusterModuleType, Dict[str, bool]] = {
    ClusterModuleType.QCM: {
        "complex_outputs": True,
        "real_outputs": True,
        "digital_outputs": True,
    },
    ClusterModuleType.QRM: {
        "complex_outputs": True,
        "complex_inputs": True,
        "real_outputs": True,
        "real_inputs": True,
        "digital_outputs": True,
    },
    ClusterModuleType.QCM_RF: {
        "complex_outputs": True,
        "digital_outputs": True,
    },
    ClusterModuleType.QRM_RF: {
        "complex_outputs": True,
        "complex_inputs": True,
        "digital_outputs": True,
    },
}

_MODULE_TYPE_VALID_CHANNEL_NAMES_MAP: Dict[ClusterModuleType, Set[str]] = {
    ClusterModuleType.QCM: set(qblox_compiler.QcmModule.static_hw_properties.valid_ios),
    ClusterModuleType.QRM: set(qblox_compiler.QrmModule.static_hw_properties.valid_ios),
    ClusterModuleType.QCM_RF: set(
        qblox_compiler.QcmRfModule.static_hw_properties.valid_ios
    ),
    ClusterModuleType.QRM_RF: set(
        qblox_compiler.QrmRfModule.static_hw_properties.valid_ios
    ),
}


class ClusterModule(ExecutorConfigItem):
    """General configration for a cluster module"""

    name: str
    instrument_type: ClusterModuleType
    complex_outputs: List[Channel] = []
    complex_inputs: List[Channel] = []
    real_outputs: List[Channel] = []
    real_inputs: List[Channel] = []
    digital_outputs: List[Channel] = []

    _channel_list_fields: Tuple[str] = (
        "complex_outputs",
        "complex_inputs",
        "real_outputs",
        "real_inputs",
        "digital_outputs",
    )

    @validator(
        "complex_outputs",
        "complex_inputs",
        "real_outputs",
        "real_inputs",
        "digital_outputs",
        allow_reuse=True,
    )
    def valid_channels_for_instrument_type(
        cls, v, values, config, field: ModelField, **kwargs
    ):
        """Each Module type can only have a given list of channel lists and names"""
        try:
            module_type = values["instrument_type"]
        except KeyError:
            return v

        # only particular channel lists for given instrument type
        valid_channels = _MODULE_TYPE_VALID_CHANNELS_MAP[module_type]
        if not valid_channels.get(field.name):
            raise ValueError(
                f"'{field.name}' are not permitted in cluster modules of type '{module_type}'"
            )

        # only particular channel names for given instrument type
        permitted_channel_names = _MODULE_TYPE_VALID_CHANNEL_NAMES_MAP[module_type]
        actual_channel_names = {item.name for item in v}
        invalid_names = actual_channel_names.difference(permitted_channel_names)

        if len(invalid_names) > 0:
            raise ValueError(
                f"invalid channel names {invalid_names} for '{field.name}' for '{module_type}'. "
                f"Permitted channel names are {permitted_channel_names}"
            )

        return v

    def to_quantify(
        self, exclude_none: bool = True, exclude: Optional[Set] = None
    ) -> Dict[str, Any]:
        """Converts this cluster module into a quantify-scheduler-compatible config

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
            channel_confs: List[Channel] = getattr(self, channel_list_field, [])
            raw_dict.update(
                {
                    channel.name: channel.to_quantify()
                    for index, channel in enumerate(channel_confs)
                }
            )

        # get rid of Enum in ref
        raw_dict["instrument_type"] = f"{self.instrument_type}"

        return raw_dict


class Cluster(ExecutorConfigItem):
    """Configuration for the cluster"""

    name: str
    ref: _ClusterRef = _ClusterRef.EXTERNAL
    is_dummy: Optional[bool] = None

    instrument_address: Optional[str] = None
    instrument_type: str = "Cluster"
    sequence_to_file: Optional[bool] = None
    modules: List[ClusterModule] = []

    @validator("modules", allow_reuse=True)
    def valid_module_names(cls, v, values, **kwargs):
        """Module names can only be of the format <cluster name>_module<n>

        Where n starts at 1 and each name is unique
        """
        cluster_name = values["name"]
        module_names = [item.name for item in v]

        # no duplicate names
        duplicates = get_duplicates(module_names)
        if len(duplicates) > 0:
            raise ValueError(
                f"cluster {cluster_name} has duplicate modules {duplicates}"
            )

        # should be of right format
        pattern = re.compile(f"^{cluster_name}_module([1-9]|\\d{2,})+$")
        invalid_names = [name for name in module_names if not pattern.match(name)]
        if len(invalid_names) > 0:
            raise ValueError(
                f"Cluster {cluster_name} has invalid module names {invalid_names}. "
                "Expected module name format of <cluster name>_module<n>; n >= 1"
            )

        return v

    def to_quantify(
        self, exclude_none: bool = True, exclude: Optional[Set] = None
    ) -> Dict[str, Any]:
        """Converts this cluster into a quantify-scheduler-compatible config

        It returns something like, where n is module number, starting from 1:
        {
            instrument_type: "Cluster",
            ref: "external",
            sequence_to_file: False,
            <cluster name>_module<n>: {...},
            <cluster name>_module<n + 1>: {...},
        }
        """
        excluded_fields = exclude if isinstance(exclude, set) else set()

        # exclude some fields that are only in place
        # to make it easier to configure the hardware
        excluded_fields.update({"name", "modules", "is_dummy"})

        raw_dict = super().to_quantify(
            exclude_none=exclude_none, exclude=excluded_fields
        )

        # add the fields that are required for the hardware config
        raw_dict.update({module.name: module.to_quantify() for module in self.modules})

        # get rid of Enum in ref
        raw_dict["ref"] = f"{self.ref}"

        return raw_dict


class _QcodesInstrumentDriver(ExecutorConfigItem):
    """Metadata about the driver of the Qcodes instrument to aid in initializing it"""

    # the import path to the driver
    import_path: str
    # the key-word arguments to be passed when initializing the driver
    kwargs: Dict[str, Any]


class GenericQcodesInstrument(ExecutorConfigItem):
    """Configuration for a generic QCoDeS instrument

    Every property name is a QCoDeS command for the instrument and the
    property value corresponds to the set value
    """

    name: str
    instrument_type: str
    instrument_driver: _QcodesInstrumentDriver

    # QCoDeS parameters are special in such a way that settable parameters accept their value as a simple function
    # call e.g. instance.frequency(3600000000)
    # https://microsoft.github.io/Qcodes/examples/15_minutes_to_QCoDeS.html#Example-of-setting-and-getting-parameters
    parameters: Dict[str, Any] = {}

    def to_quantify(
        self, exclude_none: bool = True, exclude: Optional[Set] = None
    ) -> Dict[str, Any]:
        """Converts this into a quantify-scheduler-compatible config

        It spreads the parameters into a flat object

        It returns something like:
        {
            instrument_type: "LocalOscillator",
            frequency: 5e9,
            power: 20,
            IQ_state: "1",
        }
        """
        excluded_fields = exclude if isinstance(exclude, set) else set()

        # exclude some fields that are only in place
        # to make it easier to configure the hardware
        excluded_fields.update({"name", "parameters", "instrument_driver"})

        raw_dict = super().to_quantify(
            exclude_none=exclude_none, exclude=excluded_fields
        )

        # spread parameters onto the instrument's config
        raw_dict.update(self.parameters)

        return raw_dict


class SimulatorType(str, enum.Enum):
    SCQT = "scqt"
    CHALMERS = "chalmers"


class QuantifyExecutorConfig(ExecutorConfigItem):
    """The configuration constructed from the quantify hardware config JSON file"""

    backend: str = "quantify_scheduler.backends.qblox_backend.hardware_compile"
    clusters: List[Cluster] = []
    generic_qcodes_instruments: List[GenericQcodesInstrument] = []

    @classmethod
    def from_yaml(
        cls, file_path: Union[str, bytes, os.PathLike]
    ) -> "QuantifyExecutorConfig":
        """Creates a configuration from a YAML file

        Args:
            file_path: the path to the YAML file

        Raises:
            ValueError: file is not a YAML file
        """
        try:
            with open(file_path, mode="rb") as file:
                data = yaml.load(file)
                return cls(**data)

        except TypeError:
            raise ValueError(f"'{file_path}' is not a YAML file")

    def to_quantify(
        self, exclude_none: bool = True, exclude: Optional[Set] = None
    ) -> Dict[str, Any]:
        """Converts this cluster into a quantify-scheduler-compatible config

        It returns something like:
        {
            "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
            "cluster0": {
                "instrument_type":"Cluster",
                "instrument_driver":"qblox_instruments.Cluster",
                "instrument_component":"quantify_scheduler.instrument_coordinator.components.qblox.ClusterComponent",
                "sequence_to_file": "False",
                "ref": "internal",
                "cluster_module_1": {...},
                "cluster_module_2": {...},
            },
            "lo1": {"instrument_type": "LocalOscillator", "frequency": 5e9, "power": 20}
        }
        """
        excluded_fields = exclude if isinstance(exclude, set) else set()

        # exclude name as it is not part of the quantify configuration schema
        excluded_fields.update(
            {"general", "clusters", "local_oscillators", "generic_qcodes_instruments"}
        )
        return {
            **self.dict(exclude=excluded_fields, exclude_none=exclude_none),
            **{cluster.name: cluster.to_quantify() for cluster in self.clusters},
            **{
                instrument.name: instrument.to_quantify()
                for instrument in self.generic_qcodes_instruments
            },
        }
