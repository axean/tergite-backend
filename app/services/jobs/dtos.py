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
"""Data Transfer Objects for the jobs service"""
from enum import Enum, unique
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class JobStatus(str, Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    FAILED = "failed"
    SUCCESS = "successful"
    CANCELLED = "cancelled"


class TimestampPair(BaseModel):
    started: Optional[str] = None
    finished: Optional[str] = None


class JobViewTimestamps(BaseModel):
    """Timestamps for the job"""

    registration: Optional[TimestampPair] = None
    pre_processing: Optional[TimestampPair] = None
    execution: Optional[TimestampPair] = None
    post_processing: Optional[TimestampPair] = None
    final: Optional[TimestampPair] = None


class JobViewResult(BaseModel):
    """The results of the job"""

    model_config = ConfigDict(
        extra="allow",
    )

    memory: List[List[str]] = []


class JobView(BaseModel):
    """the quantum job schema that can be viewed via the BCC RESTful API"""

    model_config = ConfigDict(
        extra="allow",
    )

    job_id: str
    device: str
    calibration_date: str
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    failure_reason: Optional[str] = None
    cancellation_reason: Optional[str] = None
    timestamps: Optional[JobViewTimestamps] = None
    download_url: Optional[str] = None
    result: Optional[JobViewResult] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@unique
class LogLevel(Enum):
    """Log level of job supervisor log messages"""

    INFO = 0
    WARNING = 1
    ERROR = 2
