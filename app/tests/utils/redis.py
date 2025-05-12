# This code is part of Tergite
#
# (C) Copyright Martin Ahindura 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for redis when testing"""
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from app.tests.utils.datetime import get_current_timestamp_str

if TYPE_CHECKING:
    import redis


def insert_in_hash(
    client: "redis.Redis",
    hash_name: str,
    data: List[Dict[str, Any]],
    id_fields: Tuple[str],
):
    """Inserts the records into the redis hash map

    Args:
        client: the redis client
        hash_name: the name of the hash map to insert them into
        data: the list of records to insert
        id_fields: the name of the fields that together are used to uniquely identify a record
    """
    mapping = {
        "@@@".join([item[k] for k in id_fields]): json.dumps(item) for item in data
    }
    client.hset(name=hash_name, mapping=mapping)


def register_app_token_job_id(
    client: "redis.Redis", hash_name: str, app_token: str, job_id: str
):
    """Registers the given app token and job id

    Args:
        client: the redis client
        hash_name: the name of the hash map to insert them into
        app_token: the app token to register
        job_id: the job_id to register
    """
    redis_key = f"{job_id}@@@{app_token}"
    timestamp = get_current_timestamp_str()
    auth_log = {
        "job_id": job_id,
        "app_token": app_token,
        "status": "pending",
        "created_at": timestamp,
        "updated_at": timestamp,
    }

    client.hset(hash_name, redis_key, json.dumps(auth_log))
