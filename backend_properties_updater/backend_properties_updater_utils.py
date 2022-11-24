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


import random
from datetime import datetime, timezone

from backend_properties_updater.NDUV import NDUV

TYPE1 = "type1"
TYPE2 = "type2"
TYPE3 = "type3"
TYPE4_DOMAIN = "type4_domain"
TYPE4_CODOMAIN = "type4_codomain"
TYPE5 = "type5"


def update_NDUV(
    nduv: NDUV,
    update_rate: int,
    lower_limit: float,
    upper_limit: float,
    max_delta: float,
) -> NDUV:
    """
    Generates a new value if the time since the last update is greater than the
    specified update rate. update_rate is the time in minutes that should pass
    before the value is updated. The difference between the new and old value
    will not be greater than max_delta.
    """
    minutes_since_last_update = int(
        (datetime.now(timezone.utc) - nduv.date).total_seconds() / 60
    )

    if minutes_since_last_update < update_rate:
        return nduv

    new_value = nduv.value + random.uniform(-max_delta, max_delta)

    if new_value > upper_limit:
        new_value = upper_limit
    elif new_value < lower_limit:
        new_value = lower_limit

    return NDUV(nduv.name, datetime.now(timezone.utc), nduv.unit, new_value, nduv.types)


def init_NDUV(
    lower_limit: float, upper_limit: float, name: str, unit: str, types
) -> NDUV:
    value = random.uniform(lower_limit, upper_limit)
    return NDUV(name, datetime.now(timezone.utc), unit, value, types)
