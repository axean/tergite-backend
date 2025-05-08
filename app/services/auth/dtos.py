# This code is part of Tergite
#
# (C) Martin Ahindura 2023
# (C) Chalmers Next Labs 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Data Transfer Objects for the auth service"""
from pydantic import Field

from ...libs.store import Schema
from ...utils.datetime import get_current_timestamp
from ...utils.model import create_partial_schema
from ..jobs.dtos import JobStatus


class Credentials(Schema):
    """The model used for authenticating jobs"""

    __primary_key_fields__ = ("job_id", "app_token")

    job_id: str
    app_token: str


class AuthLog(Credentials):
    status: JobStatus
    created_at: str = Field(default_factory=get_current_timestamp)
    updated_at: str = Field(default_factory=get_current_timestamp)


# derived models
PartialAuthLog = create_partial_schema(
    "PartialAuthLog", original=AuthLog, exclude=("created_at",)
)
