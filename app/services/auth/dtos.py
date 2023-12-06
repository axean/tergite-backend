"""Data Transfer Objects for the auth service"""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    REGISTERED = "registered"
    EXECUTING = "executing"
    FAILED = "failed"
    SUCCESS = "success"


class Credentials(BaseModel):
    """The model used for authenticating jobs"""

    job_id: str
    app_token: str


class AuthLog(BaseModel):
    status: JobStatus
    created_at: datetime
    updated_at: datetime
