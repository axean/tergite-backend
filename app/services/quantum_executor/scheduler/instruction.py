# This code is part of Tergite
#
# (C) Axel Andersson (2022)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Refactored by Martin Ahindura (2024)

from typing import Dict, List
from uuid import uuid4 as uuid

import numpy as np
from qiskit.qobj import PulseQobjConfig, PulseQobjInstruction
from quantify_scheduler.enums import BinMode

from app.libs.storage_file.file import MeasLvl, MeasRet

from ..utils.general import ceil4


def meas_settings(config: PulseQobjConfig):
    setting_error = RuntimeError(
        f"Combination {(str.lower(config.meas_return), str.lower(config.meas_return))} is not supported."
    )
    if str.lower(config.meas_return) in ("avg", "average", "averaged"):
        meas_return = MeasRet(1)
        bin_mode = BinMode.AVERAGE
        if config.meas_level in (2, 1):
            acq_return_type = complex
            meas_return_cols = 1
            protocol = "SSBIntegrationComplex"
        elif config.meas_level == 0:
            acq_return_type = np.ndarray
            meas_return_cols = 16384  # length of a trace
            protocol = "trace"
        else:
            raise setting_error

    elif str.lower(config.meas_return) in ("single", "append", "appended"):
        meas_return = MeasRet(0)
        bin_mode = BinMode.APPEND
        if config.meas_level in (2, 1):
            acq_return_type = np.ndarray
            meas_return_cols = config.shots
            protocol = "SSBIntegrationComplex"
        else:
            raise setting_error
    else:
        raise setting_error

    return dict(
        acq_return_type=acq_return_type,
        protocol=protocol,
        bin_mode=bin_mode,
        meas_level=MeasLvl(config.meas_level),
        meas_return=meas_return,
        meas_return_cols=meas_return_cols,
    )


class Instruction:
    __slots__ = [
        "t0",
        "name",
        "channel",
        "port",
        "duration",
        "frequency",
        "phase",
        "memory_slot",
        "protocol",
        "parameters",
        "pulse_shape",
        "bin_mode",
        "acq_return_type",
        "label",
    ]

    def __init__(self: object, **kwargs):
        self.label = str(uuid())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __eq__(self: object, other: object) -> bool:
        self_attrs = set(
            filter(lambda attr: hasattr(self, attr), Instruction.__slots__)
        )
        other_attrs = set(
            filter(lambda attr: hasattr(other, attr), Instruction.__slots__)
        )

        # if they have different attributes, they cannot be equal
        if self_attrs != other_attrs:
            return False

        # label is always unique
        attrs = self_attrs
        attrs.remove("label")

        # if they have the same attributes, they must also all have the same values
        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False

        # otherwise, they are the same
        return True

    def __repr__(self: object) -> str:
        repr_list = [f"Instruction object @ {hex(id(self))}:"]
        for attr in Instruction.__slots__:
            if hasattr(self, attr):
                repr_list.append(f"\t{attr} : {getattr(self, attr)}".expandtabs(4))
        return "\n".join(repr_list)

    @property
    def unique_name(self: "Instruction"):
        if self.name == "parametric_pulse":
            pretty_name = self.pulse_shape
        elif self.name == "acquire":
            pretty_name = self.protocol
        else:
            pretty_name = self.name

        return f"{pretty_name}-{self.channel}-{round(self.t0 * 1e9)}"

    @classmethod
    def from_qobj(
        cls: type,
        i: PulseQobjInstruction,
        config: PulseQobjConfig,
        hardware_map: Dict[str, str] = None,
    ) -> List[object]:
        # -----------------------------------------------------------------
        if i.name == "acquire":
            attrs_list = list()
            program_settings = meas_settings(config)
            for n, qubit_idx in enumerate(i.qubits):
                attrs_list.append(
                    dict(
                        name=i.name,
                        t0=ceil4(i.t0) * 1e-9,
                        channel=f"m{qubit_idx}",
                        port=i.ch if hardware_map is None else hardware_map[f"m{qubit_idx}"],
                        duration=ceil4(i.duration) * 1e-9,
                        memory_slot=i.memory_slot[n],
                        protocol=program_settings["protocol"],
                        acq_return_type=program_settings["acq_return_type"],
                        bin_mode=program_settings["bin_mode"],
                    )
                )
        # -----------------------------------------------------------------
        elif i.name == "delay":
            attrs = dict(
                name=i.name,
                t0=ceil4(i.t0) * 1e-9,
                channel=i.ch,
                port=i.ch if hardware_map is None else hardware_map[i.ch],
                duration=ceil4(i.duration) * 1e-9,
            )
        # -----------------------------------------------------------------
        elif i.name in ("setp", "setf", "fc"):  # "shiftf" is not working
            attrs = dict(name=i.name, t0=ceil4(i.t0) * 1e-9, channel=i.ch, duration=0.0)
            if hasattr(i, "phase"):
                attrs["phase"] = i.phase
            if hasattr(i, "frequency"):
                attrs["frequency"] = i.frequency * 1e9
        # -----------------------------------------------------------------
        elif i.name == "parametric_pulse":
            attrs = dict(
                name=i.name,
                t0=ceil4(i.t0) * 1e-9,
                channel=i.ch,
                port=i.ch if hardware_map is None else hardware_map[i.ch],
                duration=ceil4(i.parameters["duration"]) * 1e-9,
                pulse_shape=i.pulse_shape,
                parameters=i.parameters,
            )
        # -----------------------------------------------------------------
        elif i.name in config.pulse_library:
            attrs = dict(
                name=i.name,
                t0=ceil4(i.t0) * 1e-9,
                channel=i.ch,
                port=i.ch if hardware_map is None else hardware_map[i.ch],
                duration=ceil4(config.pulse_library[i.name].shape[0]) * 1e-9,
            )
        # -----------------------------------------------------------------
        else:
            raise RuntimeError(f"No mapping for PulseQobjInstruction {i}")
        # -----------------------------------------------------------------
        if i.name == "acquire":
            return [Instruction(**d) for d in attrs_list]
        else:
            return [Instruction(**attrs)]


initial_object = Instruction(
    name="initial_object", t0=0.0, channel="cl0.baseband", duration=0.0
)
