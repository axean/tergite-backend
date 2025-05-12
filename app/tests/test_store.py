# This code is part of Tergite
#
# (C) Chalmers Next Labs 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
"""Module containing tests for the store library"""
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Type, Union

import pytest
from pydantic import ValidationError
from redis import Redis

from app.libs.store import Collection, ItemNotFoundError, Schema
from app.services.auth.dtos import AuthLog, PartialAuthLog
from app.tests.utils.records import with_current_timestamps

_AUTH_LOG_LIST = [
    {"job_id": "foo", "app_token": "bar"},
    {"job_id": "foot", "app_token": "bar", "status": "executing"},
    {"job_id": "feet", "app_token": "ben", "status": "failed"},
    {"job_id": "fee", "app_token": "barracuda"},
    {"job_id": "billings", "app_token": "zealot"},
]
_DELETE_SLICES = [
    (0, -1),
    (1, 4),
    (2, 3),
]


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_update_by_single_key(real_redis_client, payload, freezer):
    """Calling update() with a raw redis key updates the item if it exists already"""
    auth_logs = Collection(
        real_redis_client, schema=AuthLog, partial_schema=PartialAuthLog
    )
    payload = with_current_timestamps([payload], fields=["updated_at", "created_at"])[0]

    original_item = AuthLog(**{"status": "pending", **payload})
    _insert_into_redis(real_redis_client, [original_item])
    original_item_in_db = _get_redis_value(real_redis_client, original_item)
    key = _get_redis_key(original_item)

    new_update = {
        "status": "successful",
        "job_id": payload["job_id"],
        "app_token": payload["app_token"],
    }
    new_item = auth_logs.update(key, PartialAuthLog(**new_update))
    new_item_in_db = _get_redis_value(real_redis_client, original_item)

    assert original_item_in_db == original_item.model_dump_json()
    assert new_item_in_db == new_item.model_dump_json()
    assert new_item.model_dump() == {
        **original_item.model_dump(),
        "status": "successful",
    }


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_dict_update_by_single_key(real_redis_client, payload, freezer):
    """Calling update() with a raw redis key, and dict updates, changes the item if it exists already"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)
    payload = with_current_timestamps([payload], fields=["updated_at", "created_at"])[0]

    original_item = AuthLog(**{"status": "pending", **payload})
    _insert_into_redis(real_redis_client, [original_item])
    original_item_in_db = _get_redis_value(real_redis_client, original_item)
    key = _get_redis_key(original_item)

    new_update = {
        "status": "successful",
        "job_id": payload["job_id"],
        "app_token": payload["app_token"],
    }
    new_item = auth_logs.update(key, new_update)
    new_item_in_db = _get_redis_value(real_redis_client, original_item)

    assert original_item_in_db == original_item.model_dump_json()
    assert new_item_in_db == new_item.model_dump_json()
    assert new_item.model_dump() == {
        **original_item.model_dump(),
        "status": "successful",
    }


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_update_by_tuple_key(real_redis_client, payload, freezer):
    """Calling update() with a tuple of keys updates the item if it exists already"""
    auth_logs = Collection(
        real_redis_client, schema=AuthLog, partial_schema=PartialAuthLog
    )
    payload = with_current_timestamps([payload], fields=["updated_at", "created_at"])[0]

    original_item = AuthLog(**{"status": "pending", **payload})
    _insert_into_redis(real_redis_client, [original_item])
    original_item_in_db = _get_redis_value(real_redis_client, original_item)
    key = (payload["job_id"], payload["app_token"])

    new_update = {
        "status": "successful",
        "job_id": payload["job_id"],
        "app_token": payload["app_token"],
    }
    new_item = auth_logs.update(key, PartialAuthLog(**new_update))
    new_item_in_db = _get_redis_value(real_redis_client, original_item)

    assert original_item_in_db == original_item.model_dump_json()
    assert new_item_in_db == new_item.model_dump_json()
    assert new_item.model_dump() == {
        **original_item.model_dump(),
        "status": "successful",
    }


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_dict_update_by_tuple_key(real_redis_client, payload, freezer):
    """Calling update() with a tuple of keys, and dict updates, changes the item if it exists already"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)
    payload = with_current_timestamps([payload], fields=["updated_at", "created_at"])[0]

    original_item = AuthLog(**{"status": "pending", **payload})
    _insert_into_redis(real_redis_client, [original_item])
    original_item_in_db = _get_redis_value(real_redis_client, original_item)
    key = (payload["job_id"], payload["app_token"])

    new_update = {
        "status": "successful",
        "job_id": payload["job_id"],
        "app_token": payload["app_token"],
    }
    new_item = auth_logs.update(key, new_update)
    new_item_in_db = _get_redis_value(real_redis_client, original_item)

    assert original_item_in_db == original_item.model_dump_json()
    assert new_item_in_db == new_item.model_dump_json()
    assert new_item.model_dump() == {
        **original_item.model_dump(),
        "status": "successful",
    }


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_update_by_dict_key(real_redis_client, payload, freezer):
    """Calling update() with a primary key in dict form updates the item if it exists already"""
    auth_logs = Collection(
        real_redis_client, schema=AuthLog, partial_schema=PartialAuthLog
    )
    payload = with_current_timestamps([payload], fields=["updated_at", "created_at"])[0]

    original_item = AuthLog(**{"status": "pending", **payload})
    _insert_into_redis(real_redis_client, [original_item])
    original_item_in_db = _get_redis_value(real_redis_client, original_item)
    key = {
        "job_id": payload["job_id"],
        "app_token": payload["app_token"],
    }

    new_update = {"status": "successful", **key}
    new_item = auth_logs.update(key, PartialAuthLog(**new_update))
    new_item_in_db = _get_redis_value(real_redis_client, original_item)

    assert original_item_in_db == original_item.model_dump_json()
    assert new_item_in_db == new_item.model_dump_json()
    assert new_item.model_dump() == {
        **original_item.model_dump(),
        "status": "successful",
    }


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_dict_update_by_dict_key(real_redis_client, payload, freezer):
    """Calling update() with a primary key in dict form, and update dict, changes the item if it exists already"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)
    payload = with_current_timestamps([payload], fields=["updated_at", "created_at"])[0]

    original_item = AuthLog(**{"status": "pending", **payload})
    _insert_into_redis(real_redis_client, [original_item])
    original_item_in_db = _get_redis_value(real_redis_client, original_item)
    key = {
        "job_id": payload["job_id"],
        "app_token": payload["app_token"],
    }

    new_update = {"status": "successful", **key}
    new_item = auth_logs.update(key, new_update)
    new_item_in_db = _get_redis_value(real_redis_client, original_item)

    assert original_item_in_db == original_item.model_dump_json()
    assert new_item_in_db == new_item.model_dump_json()
    assert new_item.model_dump() == {
        **original_item.model_dump(),
        "status": "successful",
    }


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_update_not_found(real_redis_client, payload, freezer):
    """Calling update() fails if the item does not exist"""
    auth_logs = Collection(
        real_redis_client, schema=AuthLog, partial_schema=PartialAuthLog
    )

    key_tuple = (payload["job_id"], payload["app_token"])
    single_key = "@@@".join(key_tuple)
    key_dict = {"app_token": payload["app_token"], "job_id": payload["job_id"]}

    new_update = {"status": "successful", **key_dict}

    with pytest.raises(ItemNotFoundError, match=r"not found"):
        auth_logs.update(single_key, PartialAuthLog(**new_update))

    with pytest.raises(ItemNotFoundError, match=r"not found"):
        auth_logs.update(single_key, new_update)

    with pytest.raises(ItemNotFoundError, match=r"not found"):
        auth_logs.update(key_tuple, PartialAuthLog(**new_update))

    with pytest.raises(ItemNotFoundError, match=r"not found"):
        auth_logs.update(key_tuple, new_update)

    with pytest.raises(ItemNotFoundError, match=r"not found"):
        auth_logs.update(key_dict, PartialAuthLog(**new_update))

    with pytest.raises(ItemNotFoundError, match=r"not found"):
        auth_logs.update(key_dict, new_update)

    hmap = _get_redis_hmap(real_redis_client, schema=AuthLog)
    assert hmap == {}


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_update_invalid_schema(real_redis_client, payload, freezer):
    """update() fails if the update payload does not satisfy the schema"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)

    original_item = AuthLog(**{"status": "pending", **payload})
    _insert_into_redis(real_redis_client, [original_item])
    original_item_in_db = _get_redis_value(real_redis_client, original_item)

    key_tuple = (payload["job_id"], payload["app_token"])
    single_key = _get_redis_key(original_item)
    key_dict = {"app_token": payload["app_token"], "job_id": payload["job_id"]}

    update = WrongAuthLog(**{**payload, "status": 9})
    with pytest.raises(ValidationError, match=r"validation error for PartialAuthLog"):
        auth_logs.update(single_key, update)

    with pytest.raises(ValidationError, match=r"validation error for PartialAuthLog"):
        auth_logs.update(single_key, update.model_dump())

    with pytest.raises(ValidationError, match=r"validation error for PartialAuthLog"):
        auth_logs.update(key_tuple, update)

    with pytest.raises(ValidationError, match=r"validation error for PartialAuthLog"):
        auth_logs.update(key_tuple, update.model_dump())

    with pytest.raises(ValidationError, match=r"validation error for PartialAuthLog"):
        auth_logs.update(key_dict, update)

    with pytest.raises(ValidationError, match=r"validation error for PartialAuthLog"):
        auth_logs.update(key_dict, update.model_dump())

    item_in_db = _get_redis_value(real_redis_client, original_item)
    assert item_in_db == original_item_in_db


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_insert(real_redis_client, payload, freezer):
    """Calling insert() replaces the entire item with a new one"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)

    original_item = AuthLog(**{"status": "pending", **payload})
    _insert_into_redis(real_redis_client, [original_item])
    original_item_in_db = _get_redis_value(real_redis_client, original_item)

    new_item = AuthLog(**{**payload, "status": "successful", "created_at": "belle"})
    auth_logs.insert(new_item)
    new_item_in_db = _get_redis_value(real_redis_client, original_item)

    assert original_item_in_db == original_item.model_dump_json()
    assert new_item_in_db == new_item.model_dump_json()


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_insert_invalid_schema(real_redis_client, payload, freezer):
    """insert() fails if the payload passed does not satisfy the schema"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)

    original_item = AuthLog(**{"status": "pending", **payload})
    _insert_into_redis(real_redis_client, [original_item])
    original_item_in_db = _get_redis_value(real_redis_client, original_item)

    update = WrongAuthLog(**{**payload, "status": 9})
    with pytest.raises(ValidationError, match=r"validation error for AuthLog"):
        auth_logs.insert(update)

    item_in_db = _get_redis_value(real_redis_client, original_item)
    assert item_in_db == original_item_in_db


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_get_one(real_redis_client, payload, freezer):
    """Calling get_one() gets the item identified by the given key"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)

    item = AuthLog(**{"status": "pending", **payload})
    _insert_into_redis(real_redis_client, [item])
    item_in_db = _get_redis_value(real_redis_client, item)

    single_key = _get_redis_key(item)
    key_tuple = (payload["job_id"], payload["app_token"])
    key_dict = {"app_token": payload["app_token"], "job_id": payload["job_id"]}

    item_by_single_key = auth_logs.get_one(single_key)
    item_by_key_tuple = auth_logs.get_one(key_tuple)
    item_by_key_dict = auth_logs.get_one(key_dict)

    assert item_in_db == item.model_dump_json()
    assert item_by_single_key == item
    assert item_by_key_tuple == item
    assert item_by_key_dict == item


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_get_one_not_found(real_redis_client, payload, freezer):
    """Calling get_one() raises ItemNotFoundError if item is nonexistent"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)

    item = AuthLog(**{"status": "pending", **payload})

    single_key = _get_redis_key(item)
    key_tuple = (payload["job_id"], payload["app_token"])
    key_dict = {"app_token": payload["app_token"], "job_id": payload["job_id"]}

    with pytest.raises(ItemNotFoundError, match=r"not found"):
        auth_logs.get_one(single_key)

    with pytest.raises(ItemNotFoundError, match=r"not found"):
        auth_logs.get_one(key_tuple)

    with pytest.raises(ItemNotFoundError, match=r"not found"):
        auth_logs.get_one(key_dict)


@pytest.mark.parametrize("desc", [True, False])
def test_get_all(real_redis_client, desc, freezer):
    """Calling get_all() gets the items in the collection sorted by key"""
    items = [AuthLog(**{"status": "pending", **payload}) for payload in _AUTH_LOG_LIST]
    auth_logs = Collection(real_redis_client, schema=AuthLog)
    sorted_items = _sort_by_key(items, desc=desc)

    _insert_into_redis(real_redis_client, items)

    got = auth_logs.get_all(desc=desc)
    hmap = _get_redis_hmap(real_redis_client, schema=AuthLog)
    expected_hmap = {_get_redis_key(v): v.model_dump_json() for v in sorted_items}

    assert got == sorted_items
    assert hmap == expected_hmap


@pytest.mark.parametrize("bounds", _DELETE_SLICES)
def test_delete_many_by_single_keys(
    real_redis_client, bounds: Tuple[int, int], freezer
):
    """Calling delete_many() removes items with matching single key strings in the collection"""
    items = [AuthLog(**{"status": "pending", **payload}) for payload in _AUTH_LOG_LIST]
    auth_logs = Collection(real_redis_client, schema=AuthLog)
    keys_to_delete = [_get_redis_key(item) for item in items[bounds[0] : bounds[1]]]
    expected_items = items[: bounds[0]] + items[bounds[1] :]

    _insert_into_redis(real_redis_client, items)

    auth_logs.delete_many(keys_to_delete)
    hmap = _get_redis_hmap(real_redis_client, schema=AuthLog)
    expected_hmap = {_get_redis_key(v): v.model_dump_json() for v in expected_items}

    assert hmap == expected_hmap


@pytest.mark.parametrize("bounds", _DELETE_SLICES)
def test_delete_many_by_key_tuples(real_redis_client, bounds: Tuple[int, int], freezer):
    """Calling delete_many() removes items with matching key tuples in the collection"""
    items = [AuthLog(**{"status": "pending", **payload}) for payload in _AUTH_LOG_LIST]
    auth_logs = Collection(real_redis_client, schema=AuthLog)
    keys_to_delete = [
        (item.job_id, item.app_token) for item in items[bounds[0] : bounds[1]]
    ]
    expected_items = items[: bounds[0]] + items[bounds[1] :]

    _insert_into_redis(real_redis_client, items)

    auth_logs.delete_many(keys_to_delete)
    hmap = _get_redis_hmap(real_redis_client, schema=AuthLog)
    expected_hmap = {_get_redis_key(v): v.model_dump_json() for v in expected_items}

    assert hmap == expected_hmap


@pytest.mark.parametrize("bounds", _DELETE_SLICES)
def test_delete_many_by_key_dicts(real_redis_client, bounds: Tuple[int, int], freezer):
    """Calling delete_many() removes items with matching key dictionaries in the collection"""
    items = [AuthLog(**{"status": "pending", **payload}) for payload in _AUTH_LOG_LIST]
    auth_logs = Collection(real_redis_client, schema=AuthLog)
    keys_to_delete = [
        {"job_id": item.job_id, "app_token": item.app_token}
        for item in items[bounds[0] : bounds[1]]
    ]
    expected_items = items[: bounds[0]] + items[bounds[1] :]

    _insert_into_redis(real_redis_client, items)

    auth_logs.delete_many(keys_to_delete)
    hmap = _get_redis_hmap(real_redis_client, schema=AuthLog)
    expected_hmap = {_get_redis_key(v): v.model_dump_json() for v in expected_items}

    assert hmap == expected_hmap


def test_clear(real_redis_client, freezer):
    """Calling clear() removes all items in hashmap"""
    items = [AuthLog(**{"status": "pending", **payload}) for payload in _AUTH_LOG_LIST]
    auth_logs = Collection(real_redis_client, schema=AuthLog)

    _insert_into_redis(real_redis_client, items)

    auth_logs.clear()
    items_in_db = _get_redis_hmap(real_redis_client, schema=AuthLog)

    assert items_in_db == {}


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_exists(real_redis_client, payload, freezer):
    """Calling exists() checks that key exists"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)

    item = AuthLog(**{"status": "pending", **payload})

    single_key = _get_redis_key(item)
    key_tuple = (payload["job_id"], payload["app_token"])
    key_dict = {"app_token": payload["app_token"], "job_id": payload["job_id"]}

    assert not auth_logs.exists(single_key)
    assert not auth_logs.exists(key_tuple)
    assert not auth_logs.exists(key_dict)

    _insert_into_redis(real_redis_client, [item])

    assert auth_logs.exists(single_key)
    assert auth_logs.exists(key_tuple)
    assert auth_logs.exists(key_dict)


def _get_current_timestamp():
    """Gets the current timestamp"""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_redis_value(redis: Redis, item: Schema) -> Union[str, None]:
    """Get the value of the item in redis

    Args:
        redis: the redis connection
        item: the item that might have been stored in the redis

    Returns:
        the value saved in redis for the given item or None if not exists
    """
    hashmap_name = _get_redis_hashmap_name(item.__class__)
    redis_key = _get_redis_key(item)
    value: Union[bytes, None] = redis.hget(hashmap_name, redis_key)
    if isinstance(value, bytes):
        return value.decode()

    return value


def _get_redis_hmap(redis: Redis, schema: Type[Schema]) -> Dict[str, str]:
    """Get the full hashmap for the given schema in redis sorted by key

    Args:
        redis: the redis connection
        schema: the schema whose collection is to be queried

    Returns:
        the dictionary of keys and values in the redis hashmap of the schema
    """
    hashmap_name = _get_redis_hashmap_name(schema)
    raw_value: Dict[bytes, bytes] = redis.hgetall(hashmap_name)
    return {k.decode(): v.decode() for k, v in raw_value.items()}


def _insert_into_redis(redis: Redis, items: List[Schema]):
    """Inserts the items into redis

    Args:
        redis: the redis connection
        items: the items that are to be inserted
    """
    try:
        hashmap_name = _get_redis_hashmap_name(items[0].__class__)
    except IndexError:
        return

    redis_items = {_get_redis_key(item): item.model_dump_json() for item in items}
    redis.hset(hashmap_name, mapping=redis_items)


def _get_redis_hashmap_name(schema: Type[Schema]) -> Union[str, None]:
    """Get the name of the hashmap for the given schema

    Args:
        schema: the schema under consideration

    Returns:
        the name of the hashmap where the item is stored
    """
    return f"{schema.__module__}.{schema.__qualname__}".lower()


def _get_redis_key(item: Schema) -> Union[str, None]:
    """Get the key of the item in redis

    Args:
        item: the item under consideration

    Returns:
        the key for the given item
    """
    keys = tuple(
        getattr(item, field) for field in item.__class__.__primary_key_fields__
    )
    return "@@@".join(keys)


def _sort_by_key(items: List[Schema], desc: bool = False) -> List[Schema]:
    """Sorts the items by the key

    Args:
        items: the items to sort
        desc: whether to return in ascending or descending order

    Returns:
        the items sorted
    """
    return sorted(items, key=lambda v: _get_redis_key(v), reverse=desc)


class WrongAuthLog(Schema):
    __primary_key_fields__ = ("job_id", "app_token")

    job_id: str
    app_token: str
    status: int
