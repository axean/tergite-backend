# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import uuid


def validate_uuid_str(id_str, version):
    try:
        temp_uuid = uuid.UUID(id_str, version=version)
    except (ValueError, TypeError):
        return False
    return str(temp_uuid) == id_str


def validate_uuid4_str(id_str):
    return validate_uuid_str(id_str, 4)
