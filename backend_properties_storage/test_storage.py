# This code is part of Tergite
#
# (C) Copyright David Wahlstedt 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import logging

import pytest
import redis

from backend_properties_storage.storage import (
    BackendProperty,
    PropertyType,
    get_resonator_property,
    get_resonator_value,
    set_resonator_property,
    set_resonator_value,
)

"""Logging initialization"""

logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
# The following two lines are not used yet, but can be good to have available:
logger.setLevel(logging.INFO)
LOGLEVEL = logging.INFO


"""Redis initialization"""
red = redis.Redis()


"""Test fixture"""


def _del_keys():
    # All the Redis keys will have this prefix. This exposes the Redis
    # key structure, but we only do it in this test module.
    keys = red.keys(f"*test_*")
    for key in keys:
        red.delete(key)


@pytest.fixture
def fixture():
    # setup
    _del_keys()
    yield
    # teardown
    _del_keys()


"""Tests

Run them as follows:
(-s flag makes printouts visible, and  -o log_cli=true enables logging messages)
pytest -o log_cli=true -vvv -s backend_properties_config/test_load_config.py
"""

"""BackendProperty tests"""


def test_write_metadata_read(fixture):
    """Create a backend property and write its metadata to Redis. The
    count and the value should be undefined in Redis, but the
    timestamp should be set.
    """
    # property_type, name, component, and index are used to identify the record
    property_type = PropertyType.DEVICE
    name = "test_write_metadata_read"
    # These two are optional:
    component = "resonator"
    index = 0

    p = BackendProperty(
        property_type=property_type,
        name=name,
        value=50.0,
        component=component,
        index=index,
        unit="Hz",
        source="test",
    )
    logger.info(f"Before write: {p=}")
    p.write_metadata()
    q, timestamp, count = BackendProperty.read(
        property_type=property_type, name=name, component=component, index=index
    )
    logger.info(f"After read: {q=}, {timestamp=}, {count=}")
    # test that some field was actually set in Redis
    assert q.unit == p.unit
    # the following should not be set by write_metadata:
    assert q.value is None
    assert count is None


def test_write_value_read(fixture):
    """Create a backend property and write its value to Redis. The
    value should be set in Redis, the counter should be incremented,
    and the timestamp should be updated.
    """
    property_type = PropertyType.DEVICE
    name = "test_write_value_read"
    # These two are optional:
    component = "resonator"
    index = 0

    p = BackendProperty(
        property_type=property_type,
        name=name,
        value=50.0,
        component=component,
        index=index,
        unit="Hz",
        source="test",
    )
    logger.info(f"Before write: {p=}")
    p.write_value()
    q, timestamp, count = BackendProperty.read(
        property_type=property_type, name=name, component=component, index=index
    )
    logger.info(f"After read: {q=}, {timestamp=}, {count=}")
    assert q.value == 50.0
    assert count == 1


def test_write_value_read_no_component(fixture):
    """Same as test_write_value_read, but with no component or index,
    just to have a test that doesn't use component and index
    """
    property_type = PropertyType.DEVICE
    name = "test_write_value_read_no_component"

    p = BackendProperty(
        property_type=property_type,
        name=name,
        value=50.0,
        unit="Hz",
        source="test",
    )
    logger.info(f"Before write: {p=}")
    p.write_value()
    q, timestamp, count = BackendProperty.read(property_type=property_type, name=name)
    logger.info(f"After read: {q=}, {timestamp=}, {count=}")
    assert q.value == 50.0
    assert count == 1


def test_write_value_twice(fixture):
    # Write twice and check that the increment is 2.
    property_type = PropertyType.DEVICE
    name = "test_write_value_twice"
    # These two are optional:
    component = "resonator"
    index = 0

    p = BackendProperty(
        property_type=property_type,
        name=name,
        value=50.0,
        component=component,
        index=index,
        unit="Hz",
        source="test",
    )
    logger.info(f"Before write: {p=}")
    p.write_metadata()
    count_1 = BackendProperty.get_counter(
        property_type=property_type,
        name=name,
        component=component,
        index=index,
    )
    assert count_1 == 0  # no counter increase for metadata
    p.write_value()
    count_2 = BackendProperty.get_counter(
        property_type=property_type,
        name=name,
        component=component,
        index=index,
    )
    assert count_2 == 1
    p.write_value()
    count_3 = BackendProperty.get_counter(
        property_type=property_type,
        name=name,
        component=component,
        index=index,
    )
    assert count_3 == 2


def test_reset_counter(fixture):
    property_type = PropertyType.DEVICE
    name = "test_reset_counter"
    # These two are optional:
    component = "resonator"
    index = 0

    p = BackendProperty(
        property_type=property_type,
        name=name,
        value=50.0,
        component=component,
        index=index,
        unit="Hz",
        source="test",
    )
    p.write_metadata()
    for _ in range(5):
        p.write_value()

    count = BackendProperty.get_counter(
        property_type=property_type,
        name=name,
        component=component,
        index=index,
    )

    assert count == 5
    BackendProperty.reset_counter(
        property_type=property_type,
        name=name,
        component=component,
        index=index,
    )
    count_2 = BackendProperty.get_counter(
        property_type=property_type,
        name=name,
        component=component,
        index=index,
    )
    assert count_2 == 0


def test_get_timestamp(fixture):
    property_type = PropertyType.DEVICE
    name = "test_get_timestamp"
    # These two are optional:
    component = "resonator"
    index = 0

    p = BackendProperty(
        property_type=property_type,
        name=name,
        value=50.0,
        component=component,
        index=index,
        unit="Hz",
        source="test",
    )
    p.write_metadata()

    timestamp = BackendProperty.get_timestamp(
        property_type=property_type,
        name=name,
        component=component,
        index=index,
    )
    logger.info(f"{timestamp=}")


def test_get_timestamp_unknown(fixture):
    """Get the timestamp of a property that has not been written into
    Redis. Result should be None
    """
    property_type = PropertyType.DEVICE
    name = "test_get_timestamp_unknown"
    # These two are optional:
    component = "resonator"
    index = 0

    timestamp = BackendProperty.get_timestamp(
        property_type=property_type,
        name=name,
        component=component,
        index=index,
    )
    logger.info(f"{timestamp=}")
    assert timestamp is None


def test_read_unknown(fixture):
    """Read a property that has not been written into Redis"""
    property_type = PropertyType.DEVICE
    name = "test_read_unknown"
    # These two are optional:
    component = "resonator"
    index = 0
    result = BackendProperty.read(
        property_type=property_type, name=name, component=component, index=index
    )
    logger.info(f"After read: {result=}")
    assert result == None


def test_read_value_unknown(fixture):
    property_type = PropertyType.DEVICE
    name = "test_read_value_unknown"
    # These two are optional:
    component = "resonator"
    index = 0

    p = BackendProperty(
        property_type=property_type,
        name=name,
        value=50.0,
        component=component,
        index=index,
        unit="Hz",
        source="test",
    )
    logger.info(f"Before write_metadata: {p=}")
    p.write_metadata()
    # at this point, value is not set
    result = BackendProperty.read_value(
        property_type, name, component=component, index=index
    )

    logger.info(f"After read_field: {result=}")
    # Problem: how do we differentiate from when the key is not set,
    # and when it's actually set to 'None'?  The type Optional[T],
    # when the value is None is ambiguous. In Rust we could have told
    # the difference between Some(None) and None, but in Python we
    # can't.
    #
    # If we adopt the convention that metadata must always be set
    # before values, we can check the metadata to tell the difference?
    assert result == None


""" Manual test as complement:

The test below,test_set_resonator_property, has also been tested
manually together with below script:

while true; do t=$(( RANDOM % 6 )); t2="$(echo 0.7 + 0.$t |bc)"; redis-cli set "device:resonator:0:test_set_resonator_property:count" 1; echo "sleeping $t2"; sleep "$t2";done

It writes to a Redis key in random time intervals between 0.7 and 1.2 seconds.

In order to test the transaction mechanism. a 1 s sleep was inserted
in the write_value pipeline block, to let the script interrupt it. This
confirms that the method retries until there is not interruption from
other accesses. Another manual test does the script in a faster pace,
and verifies that the max retries limit is exceeded, and so None is
returned. More testing is of course needed, but at least the basic
idea seems to work.
"""


def test_set_resonator_property(fixture):
    name = "test_set_resonator_property"
    index = 0
    value = 6e9
    set_resonator_property(name, index, value=value, unit="Hz")

    result, timestamp, count = get_resonator_property(name, index)

    assert timestamp is not None
    assert count == 1
    assert result is not None
    assert result.value == value
    assert result.unit == "Hz"


def test_set_resonator_value(fixture):
    name = "test_set_resonator_value"
    index = 0
    value = 6e9
    set_resonator_value(name, index, value)

    result = get_resonator_value(name, index)

    assert result is not None
    assert result == value


def test_delete_property(fixture):
    property_type = PropertyType.DEVICE
    name = "test_read_value_unknown"
    # These two are optional:
    component = "resonator"
    index = 0
    p = BackendProperty(
        property_type=property_type,
        name=name,
        value=50.0,
        component=component,
        index=index,
        unit="Hz",
        source="test",
    )
    p.write_metadata()
    p.write_value()
    BackendProperty.delete_property(
        property_type=property_type,
        name=name,
        component=component,
        index=index,
    )
    result = BackendProperty.read(
        property_type=property_type,
        name=name,
        component=component,
        index=index,
    )
    assert result is None
