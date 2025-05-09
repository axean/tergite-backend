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
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.libs.store import Schema
from app.utils.datetime import utc_now_str

StageName = Literal[
    "registration",
    "pre_processing",
    "execution",
    "post_processing",
    "final",
]
TimestampLabel = Literal["started", "finished"]


@unique
class Stage(int, Enum):
    """Stage in the BCC chain"""

    REG_Q = 0
    REG_W = 1
    # FIXME: We will skip the preprocessing for now
    PRE_PROC_Q = 2
    PRE_PROC_W = 3
    EXEC_Q = 4
    EXEC_W = 5
    PST_PROC_Q = 6
    PST_PROC_W = 7
    FINAL_Q = 8
    FINAL_W = 9

    @property
    def verbose_name(self) -> str:
        """The name of this stage in a verbose manner"""
        return _STAGE_VERBOSE_NAME_MAP[self]


class JobStatus(str, Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    FAILED = "failed"
    SUCCESS = "successful"
    CANCELLED = "cancelled"


class TimestampPair(BaseModel):
    started: Optional[str] = None
    finished: Optional[str] = None


class Timestamps(BaseModel):
    """Timestamps for the job"""

    registration: Optional[TimestampPair] = None
    pre_processing: Optional[TimestampPair] = None
    execution: Optional[TimestampPair] = None
    post_processing: Optional[TimestampPair] = None
    final: Optional[TimestampPair] = None

    def with_updates(self, updates: Dict[StageName, Dict[TimestampLabel, str]]):
        """Generates a new timestamp instance with the new partial updates

        Args:
            updates: dict of partial updates to incorporate into the new timestamp

        Returns:
            a new timestamp with the given updates
        """
        parsed_updates = self.model_validate(updates)
        updates_dict = parsed_updates.model_dump(
            exclude_unset=True, exclude_defaults=True
        )

        model_copy = self.model_copy()

        for name, new_pair in updates_dict.items():  # type: str, dict
            original_pair = getattr(model_copy, name)

            for label, timestamp in new_pair.items():
                if original_pair is None:
                    original_pair = TimestampPair()
                    setattr(model_copy, name, original_pair)

                setattr(original_pair, label, timestamp)

        return model_copy


class JobResult(BaseModel):
    """The results of the job"""

    model_config = ConfigDict(
        extra="allow",
    )

    memory: List[List[str]] = []


class Job(Schema):
    """the quantum job schema that can be viewed via the BCC RESTful API"""

    __primary_key_fields__ = ("job_id",)

    model_config = ConfigDict(
        extra="allow",
    )

    job_id: str
    device: str
    calibration_date: str
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    stage: Stage = Stage.REG_Q
    status: JobStatus = JobStatus.PENDING
    failure_reason: Optional[str] = None
    cancellation_reason: Optional[str] = None
    timestamps: Optional[Timestamps] = None
    download_url: Optional[str] = None
    result: Optional[JobResult] = None
    created_at: Optional[str] = Field(default_factory=utc_now_str)
    updated_at: Optional[str] = Field(default_factory=utc_now_str)


@unique
class LogLevel(Enum):
    """Log level of job supervisor log messages"""

    INFO = 0
    WARNING = 1
    ERROR = 2


_STAGE_VERBOSE_NAME_MAP: Dict[Stage, str] = {
    Stage.REG_Q: "registration queue",
    Stage.REG_W: "registration worker",
    Stage.PRE_PROC_Q: "pre-processing queue",
    Stage.PRE_PROC_W: "pre-processing worker",
    Stage.EXEC_Q: "execution queue",
    Stage.EXEC_W: "execution worker",
    Stage.PST_PROC_Q: "post-processing queue",
    Stage.PST_PROC_W: "post-processing worker",
    Stage.FINAL_Q: "finalization queue",
    Stage.FINAL_W: "finalization worker",
}
