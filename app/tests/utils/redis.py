"""Utility functions for redis"""
import json
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import redis


def insert_in_hash(
    client: "redis.Redis", hash_name: str, data: List[Dict[str, Any]], id_field: str
):
    """Inserts the records into the redis hash map

    Args:
        client: the redis client
        hash_name: the name of the hash map to insert them into
        data: the list of records to insert
        id_field: the name of the field that is unique for every record
    """
    mapping = {record[id_field]: json.dumps(record) for record in data}
    client.hset(name=hash_name, mapping=mapping)
