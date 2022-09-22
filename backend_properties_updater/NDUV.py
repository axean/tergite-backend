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


class NDUV:
    """
    Container for a name, date, unit, and value.
    """

    def __init__(
        self,
        name: str,
        date: datetime,
        unit: str,
        value: float,
        types,
    ):
        self.name = name
        self.date = date
        self.unit = unit
        self.value = value
        self.types = types

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "date": self.date.isoformat(),
            "unit": self.unit,
            "value": self.value,
            "types": self.types,
        }
