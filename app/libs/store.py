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
"""Module containing the source code for storing data"""
from typing import Any, Dict, Generic, List, Sequence, Tuple, Type, TypeVar, Union

from pydantic import BaseModel
from redis import Redis

from app.utils.exc import BaseBccException

_KEY_SEPARATOR = "@@@"


class Schema(BaseModel):
    """The base class for all schemas to be used in collections"""

    __primary_key_fields__: Tuple[str, ...] = ("id",)

    @classmethod
    def construct_redis_key(
        cls, __key: Union[str, Tuple[Any, ...], Dict[str, Any]]
    ) -> str:
        """Gets the redis key given the primary key values

        Args:
            __key: Can be the value of the primary key field,
                or the values of the primary fields in the right order,
                or a dictionary of the primary fields and their values

        Returns:
            the redis key string

        Raises:
            KeyError: some primary key fields were not set
        """
        keys = __key
        if isinstance(__key, str):
            keys = (__key,)
        if isinstance(__key, dict):
            try:
                keys = [__key[k] for k in cls.__primary_key_fields__]
            except KeyError as exp:
                raise KeyError(f"some primary key fields were not set: {exp}")

        return _KEY_SEPARATOR.join(keys)


T = TypeVar("T", bound=Schema)


class Collection(Generic[T]):
    """A synchronous collection of items of similar types"""

    def __init__(self, connection: Redis, schema: Type[T]):
        self._connection = connection
        self._schema = schema
        self._hashmap_name = f"{schema.__module__}.{schema.__qualname__}".lower()

    def exists(self, key: Union[str, Tuple[Any, ...], Dict[str, Any]]) -> bool:
        """Checks if an item with the same primary keys exists

        The key can be the raw key used in redis,
        or the values of the primary fields in the right order,
        or a dictionary of the primary fields and their values

        Args:
            key: the unique key that identifies that item

        Returns:
            True if the collection contains an item with the same key

        Raises:
            KeyError: some primary key fields were not set
        """
        redis_key = self._schema.construct_redis_key(key)
        return self._connection.hexists(self._hashmap_name, redis_key)

    def get_one(self, key: Union[str, Tuple[Any, ...], Dict[str, Any]]) -> T:
        """Get one item by key

        The key can be the raw key used in redis,
        or the values of the primary fields in the right order,
        or a dictionary of the primary fields and their values

        Args:
            key: the unique key that identifies that item

        Returns:
            the item identified by the given key in the hash

        Raises:
            ItemNotFoundError: "key '{key}' not found"
            ValidationError: item does not match the given schema
            KeyError: some primary key fields were not set
        """
        redis_key = self._schema.construct_redis_key(key)

        data = self._connection.hget(self._hashmap_name, redis_key)
        if data is None:
            raise ItemNotFoundError(f"key '{key}' not found")

        return self._schema.model_validate_json(data)

    def get_all(self, desc: bool = False) -> List[T]:
        """Get all items in this collection, sorted by key

        Args:
            desc: if the items should come in descending order of keys

        Returns:
            the list of items in this collection

        Raises:
            ValidationError: item does not match the given schema
        """
        raw_data = self._connection.hgetall(self._hashmap_name)
        data = [self._schema.model_validate_json(item) for item in raw_data.values()]
        return sorted(data, key=lambda v: _get_redis_key(self._schema, v), reverse=desc)

    def upsert(self, payload: T) -> T:
        """Updates the item identified by the primary key or inserts it if it does not exist

        The payload passed can be a partial schema, but it will result in validation
        errors if the payload does not satisfy the full schema yet the item does not exist already.

        Args:
            payload: the item to update or insert.

        Returns:
            the current item as existing in the collection

        Raises:
            ValidationError: payload does not satisfy the schema of the collection
            AttributeError: some primary key fields were not set
        """
        key = _get_redis_key(self._schema, payload)
        props = payload.model_dump(exclude_unset=True, exclude_defaults=True)

        try:
            old_item = self.get_one(key)
            props = {**old_item.model_dump(), **props}
        except ItemNotFoundError:
            pass

        new_item = self._schema(**props)
        data = new_item.model_dump_json()
        self._connection.hset(self._hashmap_name, key, data)

        return new_item

    def replace(self, payload: T):
        """Replaces the item identified by the primary key or inserts it if it does not exist

        Args:
            payload: the item to update or insert

        Returns:
            the current item in the collection

        Raises:
            ValidationError: payload does not satisfy the schema of the collection
            AttributeError: some primary key fields were not set
        """
        key = _get_redis_key(self._schema, payload)

        self._schema.model_validate(payload, from_attributes=True)
        data = payload.model_dump_json()
        self._connection.hset(self._hashmap_name, key, data)

        return payload

    def delete_many(self, keys: Sequence[Union[str, Tuple[Any, ...], Dict[str, Any]]]):
        """Get many items by their keys

        The keys can be the tuples of the raw keys used in redis,
        or sequence of tuples of the values of the primary fields in the right order,
        or sequence of dictionaries of the primary fields and their values

        Args:
            keys: the unique keys that identify that items

        Raises:
            ValidationError: item does not match the given schema
            KeyError: some primary key fields were not set
        """
        redis_keys = [self._schema.construct_redis_key(key) for key in keys]
        self._connection.hdel(self._hashmap_name, *redis_keys)

    def clear(self):
        """Clears all items in this collection"""
        self._connection.delete(self._hashmap_name)


class ItemNotFoundError(BaseBccException):
    """Exception when item is not found"""


def _get_redis_key(schema: Type[Schema], item: Any) -> str:
    """Gets the redis key for this item

    Args:
        schema: the schema under consideration
        item: the item from which to extract the key

    Returns:
        the redis key string corresponding to the given item

    Raises:
        AttributeError: some primary key fields were not set
    """
    try:
        keys = tuple(getattr(item, field) for field in schema.__primary_key_fields__)
        return _KEY_SEPARATOR.join(keys)
    except AttributeError as exp:
        raise AttributeError(f"some primary key fields were not set: {exp}")
