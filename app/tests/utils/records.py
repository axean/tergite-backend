# This code is part of Tergite
#
# (C) Copyright Martin Ahindura 2023, 2024
# (C) Copyright Chalmers Next Labs 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
"""Utility functions for records"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Sequence

from app.tests.utils.datetime import get_timestamp_str


def order_by(
    data: List[Dict[str, Any]], field: str, is_descending: bool = False
) -> List[Dict[str, Any]]:
    """Orders the data by given field

    Args:
        data: the list of records to sort
        field: the field to order by
        is_descending: whether to sort in descending order

    Returns:
        the ordered list of records
    """
    return sorted(data, key=lambda x: x[field], reverse=is_descending)


def with_incremental_timestamps(
    data: List[dict],
    fields: Sequence[str] = (
        "created_at",
        "updated_at",
    ),
) -> List[dict]:
    """Gets data that has timestamps, each record with an earlier timestamp than the next

    We update the fields passed with the corresponding timestamps

    Args:
        data: the list of dicts to attach timestamps to
        fields: the fields that should have the timestamps

    Returns:
        the data with timestamps
    """
    now = datetime.now(timezone.utc)
    return [
        {
            **item,
            **{
                field: get_timestamp_str(now + timedelta(minutes=idx))
                for field in fields
            },
        }
        for idx, item in enumerate(data)
    ]


def with_current_timestamps(
    data: List[dict],
    fields: Sequence[str] = ("updated_at",),
) -> List[dict]:
    """Gets data that has the current timestamp

    We update the fields passed with the corresponding timestamps

    Args:
        data: the list of dicts to attach timestamps to
        fields: the fields that should have the timestamps

    Returns:
        the data with timestamps
    """
    now = datetime.now(timezone.utc)
    return [
        {
            **item,
            **{field: get_timestamp_str(now) for field in fields},
        }
        for idx, item in enumerate(data)
    ]
