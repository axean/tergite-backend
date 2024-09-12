# This code is part of Tergite
#
# (C) Copyright Pontus Vikstål (2024)
# (C) Copyright Adilet Tuleouv (2024)
# (C) Copyright Stefan Hill (2024)
# (C) Copyright Martin Ahindura (2024)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from datetime import datetime
from os import PathLike
from typing import Optional, Dict, Any, List, Tuple, Union, Literal

import toml
from pydantic import BaseModel, Extra

from app.libs.properties.utils.date_time import utc_to_iso, utc_now_iso


class _QubitProps:
    """Qubit Device configuration"""

    frequency: int
    pi_pulse_amplitude: float
    pi_pulse_duration: float
    pulse_type: str
    pulse_sigma: float
    t1_decoherence: float
    t2_decoherence: float
    id: Optional[int] = None
    index: Optional[int] = None
    x_position: Optional[int] = None
    y_position: Optional[int] = None
    xy_drive_line: Optional[int] = None
    z_drive_line: Optional[int] = None


class _ReadoutResonatorProps(BaseModel):
    """ReadoutResonator Device configuration"""

    acq_delay: float
    acq_integration_time: float
    frequency: int
    pulse_amplitude: float
    pulse_delay: float
    pulse_duration: float
    pulse_type: str
    id: Optional[int] = None
    index: Optional[int] = None
    x_position: Optional[int] = None
    y_position: Optional[int] = None
    readout_line: Optional[int] = None
    lda_parameters: Optional[Dict[str, Any]] = None


class _DeviceProperties(BaseModel):
    """All Device Properties"""

    qubit: Optional[List[_QubitProps]] = None
    readout_resonator: Optional[List[_ReadoutResonatorProps]] = None
    coupler: Optional[List[Dict[str, Any]]] = None

    class Config:
        arbitrary_types_allowed = True



class DeviceV1(BaseModel):
    """Basic structure of the config of a device"""

    name: str
    characterized: bool
    open_pulse: bool
    version: str
    meas_map: List[List[int]]
    coupling_map: List[Tuple[int, int]]
    description: str = None
    simulator: bool = False
    num_qubits: int = 0
    num_couplers: int = 0
    num_resonators: int = 0
    online_date: Optional[str] = None
    dt: Optional[float] = None
    dtm: Optional[float] = None
    timelog: Dict[str, Any] = {}
    qubit_ids: Dict[int, str] = {}
    device_properties: Optional[_DeviceProperties] = None
    discriminators: Optional[Dict[str, Any]] = None
    meas_lo_freq: Optional[List[int]] = None
    qubit_lo_freq: Optional[List[int]] = None
    qubit_calibrations: Optional[Dict[str, Any]] = None
    coupler_calibrations: Optional[Dict[str, Any]] = None
    resonator_calibrations: Optional[Dict[str, Any]] = None
    gates: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    properties: Optional[Dict[str, Any]] = None

    class Config:
        extra = Extra.allow


class DeviceV2(BaseModel):
    """The schema for v2 of device"""

    name: str
    version: str
    number_of_qubits: int
    last_online: Optional[str] = None
    is_online: bool
    basis_gates: List[str]
    coupling_map: List[Tuple[int, int]]
    coordinates: List[Tuple[int, int]]
    is_simulator: bool

    class Config:
        extra = Extra.allow


class CalibrationValue(BaseModel, extra=Extra.allow):
    """A calibration value"""

    value: Union[float, str]
    date: Optional[str] = None
    unit: str = ""


class QubitCalibration(BaseModel, extra=Extra.allow):
    """Schema for the calibration data of the qubit"""

    t1_decoherence: Optional[CalibrationValue] = None
    t2_decoherence: Optional[CalibrationValue] = None
    frequency: Optional[CalibrationValue] = None
    anharmonicity: Optional[CalibrationValue] = None
    readout_assignment_error: Optional[CalibrationValue] = None


class DeviceCalibrationV2(BaseModel):
    """Schema for the calibration data of a given device"""

    name: str
    version: str
    qubits: List[QubitCalibration]
    last_calibrated: datetime

    class Config:
        json_encoders = {
            datetime: utc_to_iso,
        }


class _BackendGeneralConfig(BaseModel):
    """The basic config of the backend"""

    name: str
    num_qubits: int
    num_couplers: int
    num_resonators: int
    dt: float
    dtm: float
    description: str = ""
    is_active: bool = True
    characterized: bool = True
    open_pulse: bool = True
    simulator: bool = False
    version: str = "0.0.0"
    online_date: str = utc_now_iso()


class _BackendDeviceConfig(BaseModel):
    """The device config for the backend"""

    qubit_ids: List[str]
    discriminators: List[str] = ["lda", "thresholded_acquisition"]
    # the bidirectional coupling i.e. 1-to-2 coupling is represented by two tuples [1, 2], [2, 1]
    coupling_map: List[Tuple[int, int]] = []
    # the [x, y] coordinates of the qubits
    coordinates: List[Tuple[int, int]] = []
    meas_map: List[List[int]] = []
    qubit_parameters: List[str] = []
    resonator_parameters: List[str] = []
    coupler_parameters: List[str] = []
    discriminator_parameters: Dict[str, List[str]] = {}


class _BackendSimulatorConfig(BaseModel):
    """The device config for the simulated or dummy backends"""

     # Adjusted the type hint for units to support nested structure within discriminators
    units: Dict[
        Literal["qubit", "readout_resonator", "discriminators"], 
        Dict[str, str]
    ] = {}
    qubit: List[Dict[str, Union[float, str]]] = []
    readout_resonator: List[Dict[str, Union[float, str]]] = []
    discriminators: Dict[str, Dict[str, Dict[str, Union[float, str]]]] = {}


class BackendConfig(BaseModel):
    """The configration as read from the file"""

    general_config: _BackendGeneralConfig
    device_config: _BackendDeviceConfig
    gates: Dict[str, Dict[str, Any]] = {}
    simulator_config: _BackendSimulatorConfig = _BackendSimulatorConfig()
    
    @classmethod
    def from_toml(cls, file: PathLike):
        """Creates a BackendConfig instance from a TOML file"""
        with open(file, "r") as f:
            data = toml.load(f)
        return cls(
            general_config=_BackendGeneralConfig(**data["general_config"]),
            device_config=_BackendDeviceConfig(**data["device_config"]),
            gates={k: {**v} for k, v in data["gates"].items()},
            simulator_config=_BackendSimulatorConfig(**data["simulator_config"]),
        )
