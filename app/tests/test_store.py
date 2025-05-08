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
from typing import Union

import pytest
from pydantic import ValidationError
from redis import Redis

from app.libs.store import Collection, ItemNotFoundError, Schema
from app.services.auth.dtos import AuthLog, Credentials, PartialAuthLog

_AUTH_LOG_LIST = [
    {"job_id": "foo", "app_token": "bar"},
    {"job_id": "foo", "app_token": "bar", "status": "executing"},
    {"job_id": "foo", "app_token": "bar", "status": "failed"},
    {"job_id": "fee", "app_token": "barracuda"},
    {"job_id": "billings", "app_token": "zealot"},
]


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_insert(real_redis_client, payload, freezer):
    """Calling upsert() inserts the new item if it does not exist"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)
    kwargs = {"status": "pending", **payload}
    item = AuthLog(**kwargs)

    got = auth_logs.upsert(item)
    item_in_db = _get_redis_value(real_redis_client, item)

    assert got == item
    assert item_in_db == item.model_dump_json()


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_insert_invalid_schema(real_redis_client, payload, freezer):
    """upsert() fails if the new item does not satisfy the schema"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)
    item = Credentials(**payload)

    with pytest.raises(ValidationError, match=r"validation error for AuthLog"):
        auth_logs.upsert(item)

    item_in_db = _get_redis_value(real_redis_client, item)
    assert item_in_db is None


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_update(real_redis_client, payload, freezer):
    """Calling upsert() updates the item if it exists already"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)

    original_item = AuthLog(**{"status": "pending", **payload})
    _insert_into_redis(real_redis_client, original_item)
    original_item_in_db = _get_redis_value(real_redis_client, original_item)

    new_update = {
        "status": "successful",
        "job_id": payload["job_id"],
        "app_token": payload["app_token"],
    }
    new_item = auth_logs.upsert(PartialAuthLog(**new_update))
    new_item_in_db = _get_redis_value(real_redis_client, original_item)

    assert original_item_in_db == original_item.model_dump_json()
    assert new_item_in_db == new_item.model_dump_json()
    assert new_item.model_dump() == {
        **original_item.model_dump(),
        "status": "successful",
    }


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_update_invalid_schema(real_redis_client, payload, freezer):
    """upsert() fails if the update payload does not satisfy the schema"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)

    original_item = AuthLog(**{"status": "pending", **payload})
    _insert_into_redis(real_redis_client, original_item)
    original_item_in_db = _get_redis_value(real_redis_client, original_item)

    update = WrongAuthLog(**{**payload, "status": 9})
    with pytest.raises(ValidationError, match=r"validation error for AuthLog"):
        auth_logs.upsert(update)

    item_in_db = _get_redis_value(real_redis_client, original_item)
    assert item_in_db == original_item_in_db


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_replace(real_redis_client, payload, freezer):
    """Calling replace() replaces the entire item with a new one"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)

    original_item = AuthLog(**{"status": "pending", **payload})
    _insert_into_redis(real_redis_client, original_item)
    original_item_in_db = _get_redis_value(real_redis_client, original_item)

    new_item = AuthLog(**{**payload, "status": "successful", "created_at": "belle"})
    auth_logs.replace(new_item)
    new_item_in_db = _get_redis_value(real_redis_client, original_item)

    assert original_item_in_db == original_item.model_dump_json()
    assert new_item_in_db == new_item.model_dump_json()


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_replace_invalid_schema(real_redis_client, payload, freezer):
    """replace() fails if the payload passed does not satisfy the schema"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)

    original_item = AuthLog(**{"status": "pending", **payload})
    _insert_into_redis(real_redis_client, original_item)
    original_item_in_db = _get_redis_value(real_redis_client, original_item)

    update = WrongAuthLog(**{**payload, "status": 9})
    with pytest.raises(ValidationError, match=r"validation error for AuthLog"):
        auth_logs.replace(update)

    item_in_db = _get_redis_value(real_redis_client, original_item)
    assert item_in_db == original_item_in_db


@pytest.mark.parametrize("payload", _AUTH_LOG_LIST)
def test_get_one(real_redis_client, payload, freezer):
    """Calling get_one() gets the item identified by the given key"""
    auth_logs = Collection(real_redis_client, schema=AuthLog)

    item = AuthLog(**{"status": "pending", **payload})
    _insert_into_redis(real_redis_client, item)
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

    _insert_into_redis(real_redis_client, item)

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
    hashmap_name = _get_redis_hashmap_name(item)
    redis_key = _get_redis_key(item)
    value: Union[bytes, None] = redis.hget(hashmap_name, redis_key)
    if isinstance(value, bytes):
        return value.decode()

    return value


def _insert_into_redis(redis: Redis, item: Schema):
    """Inserts the value into redis

    Args:
        redis: the redis connection
        item: the item that is to be inserted
    """
    hashmap_name = _get_redis_hashmap_name(item)
    redis_key = _get_redis_key(item)
    redis.hset(hashmap_name, redis_key, item.model_dump_json())


def _get_redis_hashmap_name(item: Schema) -> Union[str, None]:
    """Get the name of the hashmap where the item would be stored in redis

    Args:
        item: the item under consideration

    Returns:
        the name of the hashmap where the item is stored
    """
    return f"{item.__class__.__module__}.{item.__class__.__qualname__}".lower()


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


class WrongAuthLog(Schema):
    __primary_key_fields__ = ("job_id", "app_token")

    job_id: str
    app_token: str
    status: int
