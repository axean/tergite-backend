# This code is part of Tergite
#
# (C) Chalmers Next Labs (2025)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import enum
import json
import os
import re
from typing import Dict, Optional, Union

from pydantic import BaseModel, Field, RootModel, field_validator
from quantify_scheduler.backends.qblox_backend import QbloxHardwareCompilationConfig

# --- Constants and Allowed Types ---

ALLOWED_TOP_LEVEL_INSTRUMENTS = {
    "Cluster",
    "LocalOscillator",
    "IQMixer",
    "OpticalModulator",
}
ALLOWED_MODULE_TYPES = {"QCM", "QRM", "QCM_RF", "QRM_RF", "QTM"}

# Regex pattern for cluster names
CLUSTER_NAME_REGEX = re.compile(r"^cluster[A-Za-z0-9]+$", re.IGNORECASE)

# --- Module and Instrument Models ---


class ClusterModuleType(str, enum.Enum):
    """Types of cluster modules"""

    QCM = "QCM"
    QRM = "QRM"
    QCM_RF = "QCM_RF"
    QRM_RF = "QRM_RF"


class ModuleConfig(BaseModel):
    instrument_type: str = Field(..., description="Module instrument type.")
    # Additional module-specific fields can be provided.

    class Config:
        extra = "allow"

    @field_validator("instrument_type")
    def validate_module_instrument_type(cls, v):
        if v not in ALLOWED_MODULE_TYPES:
            raise ValueError(
                f"Invalid module instrument_type '{v}'. Allowed types: {ALLOWED_MODULE_TYPES}"
            )
        return v


class InstrumentConfig(BaseModel):
    """
    Represents a top-level instrument configuration.
    For instrument_type 'Cluster', additional fields 'ip_address' and 'is_dummy'
    are expected along with an optional 'modules' dictionary.
    """

    instrument_type: str = Field(..., description="Top-level instrument type.")
    ref: Optional[str] = None
    modules: Optional[Dict[str, ModuleConfig]] = None

    # New fields for Cluster instruments
    ip_address: Optional[str] = Field(
        None, description="IP address for a Cluster instrument."
    )
    is_dummy: Optional[bool] = Field(
        None, description="Indicates if the cluster is a dummy cluster."
    )

    class Config:
        extra = "allow"

    @field_validator("instrument_type")
    def validate_instrument_type(cls, v):
        if v not in ALLOWED_TOP_LEVEL_INSTRUMENTS:
            raise ValueError(
                f"Invalid top-level instrument_type '{v}'. Allowed types: {ALLOWED_TOP_LEVEL_INSTRUMENTS}"
            )
        return v


# --- Main Configuration Model using RootModel ---


class QuantifyExecutorConfig(RootModel[Dict[str, InstrumentConfig]]):
    """
    Represents the entire configuration file, which is now a mapping of instrument names
    (e.g., "cluster0") to their respective configurations.
    """

    @field_validator("root")
    def validate_hardware_description(
        cls, v: Dict[str, InstrumentConfig]
    ) -> Dict[str, InstrumentConfig]:
        # Validate that for each Cluster instrument the name matches the expected pattern.
        for name, instr in v.items():
            if instr.instrument_type == "Cluster":
                if not CLUSTER_NAME_REGEX.match(name):
                    raise ValueError(
                        f"Cluster name '{name}' does not match expected pattern 'cluster<number>'."
                    )
        return v

    @classmethod
    def from_yaml(
        cls, yaml_file: Union[str, bytes, os.PathLike]
    ) -> "QuantifyExecutorConfig":
        import yaml

        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def from_json(
        cls, json_file: Union[str, bytes, os.PathLike]
    ) -> "QuantifyExecutorConfig":
        with open(json_file, "r") as f:
            data = json.load(f)

        hardware_config = QbloxHardwareCompilationConfig.model_validate(data)
        return hardware_config
