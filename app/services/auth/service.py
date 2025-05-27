"""The service file for handling authentication"""

from typing import Optional

from redis.client import Redis

from ...utils.store import Collection, ItemNotFoundError
from ..jobs.dtos import JobStatus
from .dtos import AuthLog, Credentials
from .exc import AuthenticationError, AuthorizationError, CredentialsAlreadyExists


def save_credentials(redis_db: Redis, payload: Credentials):
    """Saves the credentials passed

    Args:
        redis_db: the redis database connection where to save the credentials
        payload: the credentials to save

    Raises:
        CredentialsAlreadyExists: job id '{payload.job_id}' already exists
    """
    auth_logs = Collection(redis_db, schema=AuthLog)

    if auth_logs.exists((payload.job_id, payload.app_token)):
        raise CredentialsAlreadyExists(f"job id '{payload.job_id}' already exists")

    auth_logs.insert(
        AuthLog(
            job_id=payload.job_id,
            app_token=payload.app_token,
            status=JobStatus.PENDING,
        )
    )


def authenticate(
    redis_db: Redis,
    credentials: Credentials,
    expected_status: Optional[JobStatus] = None,
):
    """Checks whether the given credentials are valid

    Args:
        redis_db: the redis database connection where the credentials are saved
        credentials: the credentials to authenticate
        expected_status: the status that the job should be at. If None, status does not matter

    Raises:
        AuthenticationError: job {credentials.job_id} does not exist for current user
        AuthorizationError: job {credentials.job_id} is already {auth_log.status}
    """
    auth_logs = Collection(redis_db, schema=AuthLog)

    try:
        auth_log = auth_logs.get_one((credentials.job_id, credentials.app_token))
    except ItemNotFoundError:
        raise AuthenticationError(
            f"job {credentials.job_id} does not exist for current user"
        )

    if expected_status and auth_log.status != expected_status:
        raise AuthorizationError(
            f"job {credentials.job_id} is already {auth_log.status}"
        )
