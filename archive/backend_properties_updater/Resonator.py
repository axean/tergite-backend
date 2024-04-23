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


from archive.backend_properties_updater.backend_properties_updater_utils import (
    TYPE1,
    TYPE4_CODOMAIN,
    TYPE4_DOMAIN,
    init_NDUV,
    update_NDUV,
)

UPPER_FREQUENCY_LIMIT = 8e09  # HZ
LOWER_FREQUENCY_LIMIT = 6e09  # HZ
FREQUENCY_TYPES = [TYPE1, TYPE4_DOMAIN]

UPPER_RES_FREQ_GE_LIMIT = 8e09  # HZ
LOWER_RES_FREQ_GE_LIMIT = 6e09  # HZ
RES_FREQ_GE_TYPES = [TYPE1]

UPPER_RES_FREQ_GEF_LIMIT = 8e09  # HZ
LOWER_RES_FREQ_GEF_LIMIT = 6e09  # HZ
RES_FREQ_GEF_TYPES = [TYPE1]

UPPER_Q_I_LIMIT = 3e06
LOWER_Q_I_LIMIT = 1e06
Q_I_TYPES = [TYPE1]

UPPER_Q_C_LIMIT = 1e05
LOWER_Q_C_LIMIT = 1e04
Q_C_TYPES = [TYPE1]

UPPER_KAPPA_LIMIT = 500e03  # Hz
LOWER_KAPPA_LIMIT = 200e03  # Hz
KAPPA_TYPES = [TYPE1]

UPPER_READ_LENGTH_LIMIT = 100e-09  # s
LOWER_READ_LENGTH_LIMIT = 2e-06  # s
READ_LENGTH_UPDATE_RATE = 7
READ_LENGTH_MAX_DELTA = 1e-07
READ_LENGTH_TYPES = [TYPE1, TYPE4_CODOMAIN]

UPPER_READ_AMP_LIMIT = 100e-03  # V
LOWER_READ_AMP_LIMIT = 2e-06  # V
READ_AMP_UPDATE_RATE = 8
READ_AMP_MAX_DELTA = 1e-04
READ_AMP_TYPES = [TYPE1, TYPE4_CODOMAIN]

UPPER_READ_MOD_LIMIT = 1e09  # Hz
LOWER_READ_MOD_LIMIT = 100e06  # Hz
READ_MOD_UPDATE_RATE = 9
READ_MOD_MAX_DELTA = 1e05
READ_MOD_TYPES = [TYPE1]


class Resonator:
    def __init__(self, resonator_config: dict):
        self.resonator_config = resonator_config
        self.frequency = init_NDUV(
            LOWER_FREQUENCY_LIMIT,
            UPPER_FREQUENCY_LIMIT,
            "frequency",
            "Hz",
            FREQUENCY_TYPES,
        )
        self.res_frequency_ge = init_NDUV(
            LOWER_RES_FREQ_GE_LIMIT,
            UPPER_RES_FREQ_GE_LIMIT,
            "frequency_ge",
            "Hz",
            RES_FREQ_GE_TYPES,
        )
        self.res_frequency_gef = init_NDUV(
            LOWER_RES_FREQ_GEF_LIMIT,
            UPPER_RES_FREQ_GEF_LIMIT,
            "frequency_gef",
            "Hz",
            RES_FREQ_GEF_TYPES,
        )
        self.Q_i = init_NDUV(
            LOWER_Q_I_LIMIT,
            UPPER_Q_I_LIMIT,
            "Q_i",
            "",
            Q_I_TYPES,
        )
        self.Q_c = init_NDUV(
            LOWER_Q_C_LIMIT,
            UPPER_Q_C_LIMIT,
            "Q_c",
            "",
            Q_C_TYPES,
        )
        self.kappa = init_NDUV(
            LOWER_KAPPA_LIMIT,
            UPPER_KAPPA_LIMIT,
            "kappa",
            "Hz",
            KAPPA_TYPES,
        )
        self.read_length = init_NDUV(
            LOWER_READ_LENGTH_LIMIT,
            UPPER_READ_LENGTH_LIMIT,
            "read_length",
            "s",
            READ_LENGTH_TYPES,
        )
        self.read_amp = init_NDUV(
            LOWER_READ_AMP_LIMIT,
            UPPER_READ_AMP_LIMIT,
            "read_amp",
            "V",
            READ_AMP_TYPES,
        )
        self.read_mod = init_NDUV(
            LOWER_READ_MOD_LIMIT,
            UPPER_READ_MOD_LIMIT,
            "read_mod",
            "Hz",
            READ_MOD_TYPES,
        )

    def update(self):
        self.read_length = update_NDUV(
            self.read_length,
            READ_LENGTH_UPDATE_RATE,
            LOWER_READ_LENGTH_LIMIT,
            UPPER_READ_LENGTH_LIMIT,
            READ_LENGTH_MAX_DELTA,
        )

        self.read_amp = update_NDUV(
            self.read_amp,
            READ_AMP_UPDATE_RATE,
            LOWER_READ_AMP_LIMIT,
            UPPER_READ_AMP_LIMIT,
            READ_AMP_MAX_DELTA,
        )

        self.read_mod = update_NDUV(
            self.read_mod,
            READ_MOD_UPDATE_RATE,
            LOWER_READ_MOD_LIMIT,
            UPPER_READ_MOD_LIMIT,
            READ_MOD_MAX_DELTA,
        )

    def to_dict(self) -> dict:
        static_properties = [
            self.frequency.to_dict(),
            self.res_frequency_ge.to_dict(),
            self.res_frequency_gef.to_dict(),
            self.Q_i.to_dict(),
            self.Q_c.to_dict(),
            self.kappa.to_dict(),
        ]

        dynamic_properties = [
            self.read_length.to_dict(),
            self.read_amp.to_dict(),
            self.read_mod.to_dict(),
        ]

        return {
            **self.resonator_config,
            **{
                "dynamic_properties": dynamic_properties,
                "static_properties": static_properties,
            },
        }
