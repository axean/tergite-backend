# This code is part of Tergite
#
# (C) Johan Blomberg, Gustav Grännsjö 2020
# (C) Copyright David Wahlstedt 2022, 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import ast
import asyncio
from enum import Enum
from typing import Optional, Tuple

import redis

from backend_properties_storage.storage import (
    BackendProperty,
    PropertyType,
    T,
    create_redis_key,
)
from backend_properties_storage.types import TimeStamp
from utils.representation import to_string

# Set up Redis connection
red = redis.Redis(decode_responses=True)

# Redis prefix for calibration supervisor specific storage
CALIBRATION_SUPERVISOR_PREFIX = "calibration_supervisor"

# Used by check_data to indicate outcome
class DataStatus(Enum):
    in_spec = 1
    out_of_spec = 2
    bad_data = 3


# The event will be awaited by calibration.calibration_lib.request_job.
# Then, in calibration_supervisor.handle_message, when the incoming
# message matches self.requested_job_id, the event will be set. This
# causes request_job to resume after waiting, and it can
# clear the event.
class JobDoneEvent:
    def __init__(self, event: asyncio.Event):
        self.event = event
        self.requested_job_id = None


def write_calibration_result(
    node_name: str,  # name of calibration node
    property_name: str,
    value: T,
    component: Optional[str] = None,
    component_id: Optional[str] = None,
    publish: bool = True,
    **additional_fields,
):
    """This function stores a backend property based on the given arguments, and

    1. Stores the value and timestamp with the same Redis key as for a
       "public" backend property, but prefixed with
       CALIBRATION_SUPERVISOR_PREFIX, and the calibration node name,
       associated to the calibration supervisor, for internal
       purposes. This way we can store values internally that
       otherwise would be overwritten by other preocesses or
       calibration steps.

    2. If publish == True (default), also store the property in the
       "public" Redis table for backend device properties.

    The two records will have the same timestamp, so in case of
    debugging, we know they were saved by the same occation.

    Some results from measurements performed by the calibration
    supervisor are not meant for publishing as "public" device
    properties. In this case the flag `publish` can be set to False,
    and the result will only be kept for calibration supervisor
    internal purposes.
    """

    identification = {
        "property_type": PropertyType.DEVICE,
        "name": property_name,
        "component": component,
        "component_id": component_id,
    }
    # save the property as a backend property
    p = BackendProperty(
        **identification,
        value=value,
        source="measurement",
        **additional_fields,
    )
    if publish:
        p.write()

    # Save book-keeping information about this calibration goal
    property_key = create_redis_key(**identification)
    key = f"{CALIBRATION_SUPERVISOR_PREFIX}:{node_name}:{property_key}"
    # get the actual timestamp created when p was saved:
    timestamp = BackendProperty.get_timestamp(**identification)
    red.hset(key, "value", to_string(value))
    red.hset(key, "timestamp", to_string(timestamp))


def read_calibration_result(
    node_name: str,
    property_name: str,
    component: Optional[str] = None,
    component_id: Optional[str] = None,
) -> Tuple[Optional[T], Optional[TimeStamp]]:
    """Read the value and timestamp associated with the given
    arguments, stored as calibration supervisor internal information,
    from Redis.
    """

    identification = {
        "property_type": PropertyType.DEVICE,
        "name": property_name,
        "component": component,
        "component_id": component_id,
    }
    # get book-keeping information about this calibration goal
    property_key = create_redis_key(**identification)
    key = f"{CALIBRATION_SUPERVISOR_PREFIX}:{node_name}:{property_key}"
    raw_value = red.hget(key, "value")
    raw_timestamp = red.hget(key, "timestamp")
    value = ast.literal_eval(raw_value) if raw_value is not None else None
    timestamp = ast.literal_eval(raw_timestamp) if raw_timestamp is not None else None
    return value, timestamp
