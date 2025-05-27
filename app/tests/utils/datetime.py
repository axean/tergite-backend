# This code is part of Tergite
#
# (C) Copyright Martin Ahindura 2023
# (C) Copyright Chalmers Next Labs 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Test utilities for datetime"""
from datetime import datetime, timezone


def get_timestamp_str(timestamp: datetime) -> str:
    """Converts a timestamp to a string

    Args:
        timestamp: the datetime value

    Returns:
        the timestamp as a string
    """
    return timestamp.isoformat("T", timespec="milliseconds").replace("+00:00", "Z")


def get_current_timestamp_str() -> str:
    """Gets the current timestamp as a Zulu ISO format string

    Returns:
        the timestamp as a Zulu ISO format string
    """
    timestamp = datetime.now(timezone.utc)
    return get_timestamp_str(timestamp)
