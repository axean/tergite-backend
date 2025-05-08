"""The service file for handling authentication"""
from typing import Optional, Tuple

from redis.client import Redis

from ..jobs.dtos import JobStatus
from .dtos import AuthLog, Credentials
from .exc import AuthenticationError, AuthorizationError, JobAlreadyExists

_AUTH_HASH_KEY = "auth_service"
_SEPARATOR = "@@@"


def save_credentials(redis_db: Redis, payload: Credentials):
    """Saves the credentials passed

    Args:
        redis_db: the redis database connection where to save the credentials
        payload: the credentials to save

    Raises:
        JobAlreadyExists: job id '{payload.job_id}' already exists
    """
    redis_key = _get_composite_key((payload.app_token, payload.job_id))
    if redis_db.hexists(_AUTH_HASH_KEY, redis_key):
        raise JobAlreadyExists(f"job id '{payload.job_id}' already exists")

    auth_log = AuthLog(
        job_id=payload.job_id,
        app_token=payload.app_token,
        status=JobStatus.PENDING,
    )

    redis_db.hset(_AUTH_HASH_KEY, redis_key, auth_log.json())


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
    redis_key = _get_composite_key((credentials.app_token, credentials.job_id))
    auth_log_str = redis_db.hget(_AUTH_HASH_KEY, redis_key)

    if auth_log_str is None:
        raise AuthenticationError(
            f"job {credentials.job_id} does not exist for current user"
        )

    auth_log = AuthLog.parse_raw(auth_log_str)

    if expected_status and auth_log.status != expected_status:
        raise AuthorizationError(
            f"job {credentials.job_id} is already {auth_log.status}"
        )


def _get_composite_key(keys: Tuple[str, ...]) -> str:
    """Gets a single key from a list of keys

    Args:
        keys: the list of keys from which to generate the key
    """
    return _SEPARATOR.join(keys)
