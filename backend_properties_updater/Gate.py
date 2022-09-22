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


from backend_properties_updater.backend_properties_updater_utils import (
    init_NDUV,
    update_NDUV,
    TYPE1,
    TYPE4_DOMAIN,
    TYPE4_CODOMAIN,
    TYPE3,
)
from backend_properties_updater.NDUV import NDUV

UPPER_PULSE_AMP_LIMIT = 5
LOWER_PULSE_AMP_LIMIT = 0
PULSE_AMP_TYPES = [TYPE1, TYPE4_CODOMAIN]
PULSE_AMP_UPDATE_RATE = 16
PULSE_AMP_DELTA = 1e-02

UPPER_PULSE_FREQ_LIMIT = 1e09
LOWER_PULSE_FREQ_LIMIT = -1e09
PULSE_FREQ_TYPES = [TYPE1, TYPE4_CODOMAIN]
PULSE_FREQ_UPDATE_RATE = 18
PULSE_FREQ_DELTA = 1e06

UPPER_PULSE_DRAG_LIMIT = 0.5
LOWER_PULSE_DRAG_LIMIT = -0.5
PULSE_DRAG_TYPES = [TYPE1, TYPE4_CODOMAIN]
PULSE_DRAG_UPDATE_RATE = 19
PULSE_DRAG_DELTA = 1e-03

UPPER_ONE_QUBIT_PULSE_LENGTH_LIMIT = 50e-09
LOWER_ONE_QUBIT_PULSE_LENGTH_LIMIT = 10e-09
ONE_QUBIT_PULSE_LENGTH_TYPES = [TYPE1, TYPE4_CODOMAIN]
ONE_QUBIT_PULSE_LENGTH_UPDATE_RATE = 20
ONE_QUBIT_PULSE_LENGTH_DELTA = 5e-09

UPPER_TWO_QUBIT_PULSE_LENGTH_LIMIT = 300e-09
LOWER_TWO_QUBIT_PULSE_LENGTH_LIMIT = 0
TWO_QUBIT_PULSE_LENGTH_TYPES = [TYPE1, TYPE4_CODOMAIN]
TWO_QUBIT_PULSE_LENGTH_UPDATE_RATE = 21
TWO_QUBIT_PULSE_LENGTH_DELTA = 3e-08

UPPER_PULSE_DETUNE_LIMIT = 20e06
LOWER_PULSE_DETUNE_LIMIT = -20e06
PULSE_DETUNE_TYPES = [TYPE1, TYPE4_CODOMAIN]
PULSE_DETUNE_UPDATE_RATE = 22
PULSE_DETUNE_DELTA = 2e06

UPPER_GATE_ERROR_LIMIT = 1e-02
LOWER_GATE_ERROR_LIMIT = 1e-04
GATE_ERROR_TYPES = [TYPE1, TYPE3, TYPE4_DOMAIN]
GATE_ERROR_UPDATE_RATE = 23
GATE_ERROR_DELTA = 1e-03


class Gate:
    def __init__(self, gate_config: dict):
        self.gate_config = gate_config
        self.pulse_amp = init_NDUV(
            LOWER_PULSE_AMP_LIMIT,
            UPPER_PULSE_AMP_LIMIT,
            "pulse_amp",
            "V",
            PULSE_AMP_TYPES,
        )
        self.pulse_freq = init_NDUV(
            LOWER_PULSE_FREQ_LIMIT,
            UPPER_PULSE_FREQ_LIMIT,
            "pulse_freq",
            "Hz",
            PULSE_FREQ_TYPES,
        )
        self.pulse_drag = init_NDUV(
            LOWER_PULSE_DRAG_LIMIT,
            UPPER_PULSE_DRAG_LIMIT,
            "pulse_drag",
            "",
            PULSE_DRAG_TYPES,
        )
        self.pulse_detune = init_NDUV(
            LOWER_PULSE_DETUNE_LIMIT,
            UPPER_PULSE_DETUNE_LIMIT,
            "pulse_detune",
            "Hz",
            PULSE_DETUNE_TYPES,
        )

        self.pulse_length = _init_pulse_length(gate_config)

        self.gate_err = init_NDUV(
            LOWER_GATE_ERROR_LIMIT,
            UPPER_GATE_ERROR_LIMIT,
            "gate_err",
            "Hz",
            GATE_ERROR_TYPES,
        )

    def update(self):
        self.pulse_amp = update_NDUV(
            self.pulse_amp,
            PULSE_AMP_UPDATE_RATE,
            LOWER_PULSE_AMP_LIMIT,
            UPPER_PULSE_AMP_LIMIT,
            PULSE_AMP_DELTA,
        )

        self.pulse_freq = update_NDUV(
            self.pulse_freq,
            PULSE_FREQ_UPDATE_RATE,
            LOWER_PULSE_FREQ_LIMIT,
            UPPER_PULSE_FREQ_LIMIT,
            PULSE_FREQ_DELTA,
        )

        self.pulse_drag = update_NDUV(
            self.pulse_drag,
            PULSE_DRAG_UPDATE_RATE,
            LOWER_PULSE_DRAG_LIMIT,
            UPPER_PULSE_DRAG_LIMIT,
            PULSE_DRAG_DELTA,
        )

        self.pulse_detune = update_NDUV(
            self.pulse_detune,
            PULSE_DETUNE_UPDATE_RATE,
            LOWER_PULSE_DETUNE_LIMIT,
            UPPER_PULSE_DETUNE_LIMIT,
            PULSE_DETUNE_DELTA,
        )

        self.pulse_length = _update_pulse_length(self.pulse_length, self.gate_config)

        self.gate_err = update_NDUV(
            self.gate_err,
            GATE_ERROR_UPDATE_RATE,
            LOWER_GATE_ERROR_LIMIT,
            UPPER_GATE_ERROR_LIMIT,
            GATE_ERROR_DELTA,
        )

    def to_dict(self) -> dict:
        dynamic_properties = [
            self.pulse_amp.to_dict(),
            self.pulse_freq.to_dict(),
            self.pulse_drag.to_dict(),
            self.pulse_detune.to_dict(),
            self.pulse_length.to_dict(),
            self.gate_err.to_dict(),
        ]
        return {
            **self.gate_config,
            **{
                "dynamic_properties": dynamic_properties,
                "static_properties": [],
            },
        }


def _init_pulse_length(gate_config: dict) -> NDUV:
    if len(gate_config["qubits"]) == 1:
        return init_NDUV(
            LOWER_ONE_QUBIT_PULSE_LENGTH_LIMIT,
            UPPER_ONE_QUBIT_PULSE_LENGTH_LIMIT,
            "qubit_pulse_length",
            "s",
            ONE_QUBIT_PULSE_LENGTH_TYPES,
        )
    return init_NDUV(
        LOWER_TWO_QUBIT_PULSE_LENGTH_LIMIT,
        UPPER_TWO_QUBIT_PULSE_LENGTH_LIMIT,
        "qubit_pulse_length",
        "s",
        TWO_QUBIT_PULSE_LENGTH_TYPES,
    )


def _update_pulse_length(pulse_length: NDUV, gate_config: dict):
    if len(gate_config["qubits"]) == 1:
        return update_NDUV(
            pulse_length,
            ONE_QUBIT_PULSE_LENGTH_UPDATE_RATE,
            LOWER_ONE_QUBIT_PULSE_LENGTH_LIMIT,
            UPPER_ONE_QUBIT_PULSE_LENGTH_LIMIT,
            ONE_QUBIT_PULSE_LENGTH_DELTA,
        )
    return update_NDUV(
        pulse_length,
        TWO_QUBIT_PULSE_LENGTH_UPDATE_RATE,
        LOWER_TWO_QUBIT_PULSE_LENGTH_LIMIT,
        UPPER_TWO_QUBIT_PULSE_LENGTH_LIMIT,
        TWO_QUBIT_PULSE_LENGTH_DELTA,
    )
