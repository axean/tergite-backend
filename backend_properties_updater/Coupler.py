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


from datetime import datetime

from backend_properties_updater.backend_properties_updater_utils import (
    TYPE1,
    TYPE4_CODOMAIN,
    TYPE5,
    init_NDUV,
    update_NDUV,
)
from backend_properties_updater.NDUV import NDUV

UPPER_FREQ_MAX_LIMIT = 9e9
LOWER_FREQ_MAX_LIMIT = 7e9
FREQ_MAX_TYPES = [TYPE1]

UPPER_COUPLER_V0_LIMIT = 5
LOWER_COUPLER_V0_LIMIT = 4
COUPLER_V0_TYPES = [TYPE1]

UPPER_ASYMETRY_LIMIT = 0.5
LOWER_ASYMETRY_LIMIT = 0
ASYMETRY_TYPES = [TYPE1]

UPPER_XTALK_LIMIT = 0.05
LOWER_XTALK_LIMIT = 0
XTALK_TYPES = [TYPE1, TYPE5]

BIAS_V_UPDATE_RATE = 1
BIAS_V_MAX_DELTA = 0.1
BIAS_V_TYPES = [TYPE1, TYPE4_CODOMAIN]


class Coupler:
    def __init__(self, coupler_config: dict, coupling_map):
        self.coupler_config = coupler_config
        self.x_talk = _init_xtalk(coupler_config, coupling_map)
        self.freq_max = init_NDUV(
            LOWER_FREQ_MAX_LIMIT,
            UPPER_FREQ_MAX_LIMIT,
            "freq_max",
            "Hz",
            FREQ_MAX_TYPES,
        )
        self.coupler_v0 = init_NDUV(
            LOWER_COUPLER_V0_LIMIT,
            UPPER_COUPLER_V0_LIMIT,
            "coupler_V0",
            "V",
            COUPLER_V0_TYPES,
        )
        self.asymmetry = init_NDUV(
            LOWER_ASYMETRY_LIMIT,
            UPPER_ASYMETRY_LIMIT,
            "asymetry",
            "",
            ASYMETRY_TYPES,
        )
        self.bias_V = init_NDUV(
            -self.coupler_v0.value,
            self.coupler_v0.value,
            "bias_V",
            "V",
            BIAS_V_TYPES,
        )

    def update(self):
        self.bias_V = update_NDUV(
            self.bias_V,
            BIAS_V_UPDATE_RATE,
            -self.coupler_v0.value,
            self.coupler_v0.value,
            BIAS_V_MAX_DELTA,
        )

    def to_dict(self) -> dict:
        static_properties = [
            self.freq_max.to_dict(),
            self.coupler_v0.to_dict(),
            self.asymmetry.to_dict(),
            *self.x_talk,
        ]
        dynamic_properties = [self.bias_V.to_dict()]

        return {
            **self.coupler_config,
            **{
                "static_properties": static_properties,
                "dynamic_properties": dynamic_properties,
            },
        }


def _init_xtalk(coupler_config: dict, coupling_map):
    connected_qubits = coupler_config["qubits"]

    xtalks = []

    for other_coupler in coupling_map:
        # There should be no xtalk_{i,j} if i == j.
        if other_coupler["id"] == coupler_config["id"]:
            continue

        property_name = (
            "xtalk_{" + str(coupler_config["id"]) + "," + str(other_coupler["id"]) + "}"
        )

        # Couplers with shared qubits will have a non-zero xtalk value.
        if any(q in connected_qubits for q in other_coupler["qubits"]):
            nduv = init_NDUV(
                LOWER_XTALK_LIMIT,
                UPPER_XTALK_LIMIT,
                property_name,
                "",
                XTALK_TYPES,
            ).to_dict()

        else:
            nduv = NDUV(property_name, datetime.now(), "", 0, XTALK_TYPES).to_dict()
        xtalks.append(nduv)

    return xtalks
