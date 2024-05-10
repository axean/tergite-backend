# This code is part of Tergite
#
# (C) Copyright Martin Ahindura 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Dependencies useful for the FastAPI API"""
import json
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Optional, Tuple

from fastapi import Depends, HTTPException, UploadFile, status
from fastapi.requests import Request
from redis import Redis

import settings

from ..services import auth as auth_service
from ..services.kernel import service as kernel_service
from ..utils.uuid import validate_uuid4_str
from .exc import InvalidJobIdInUploadedFileError, IpNotAllowedError

_redis_connection = Redis()
_kernel_process, _kernel_connection = kernel_service.connect()


def get_redis_connection():
    """Returns a redis connection"""
    return _redis_connection


def get_job_id_dependency(job_id_field: str):
    """Creates a job_id dependency injector

    Args:
        job_id_field: the name of the parameter or json field that contains the job_id
    """

    async def get_job_id(request: Request) -> str:
        """Returns the job_id either got from the params or from the uploaded file name

        Args:
            request: the FastAPI request object

        Returns:
            the valid job_id

        Raises:
            InvalidJobIdInUploadedFileError: f"The job does not have a valid UUID4 {job_id_field}"
        """
        try:
            return request.path_params[job_id_field]
        except KeyError:
            return await get_job_id_from_uploaded_file(
                request, job_id_field=job_id_field
            )

    return get_job_id


async def get_job_id_from_uploaded_file(
    request: Request, job_id_field: str
) -> Optional[str]:
    """Extracts job_id from the uploaded file

    Args:
        request: the FastAPI request object
        job_id_field: the name of key that has the job_id

    Returns:
        the job_id in the file or None if it is invalid or does not exist

    Raises:
        InvalidJobIdInUploadedFileError: f"The job does not have a valid UUID4 {job_id_field}"
    """
    try:
        form = await request.form()
        upload_file: UploadFile = form["upload_file"]
        job_dict = json.load(upload_file.file)

        job_id = job_dict[job_id_field]
        if validate_uuid4_str(job_id):
            return job_id
    except KeyError:
        pass

    error_message = f"The job does not have a valid UUID4 {job_id_field}"
    print(error_message)
    raise InvalidJobIdInUploadedFileError(error_message)


def get_whitelisted_ip(request: Request) -> str:
    """Returns the whitelisted IP if exists or raises a IpNotAllowedError

    Args:
        request: the current FastAPI request

    Returns:
        the whitelisted IP
    """
    try:
        return request.state.whitelisted_ip
    except AttributeError:
        raise IpNotAllowedError()


def get_valid_credentials_dep(
    expected_status: Optional[auth_service.JobStatus] = None,
    job_id_field: str = "job_id",
):
    """Returns a dependency injector that gets a valid credentials with the expected job status.

    It extracts the app_token from the headers and the job_id from the parameters
    The dependency injector raises authentication or authorization errors if no
    valid app_token and job_id pair is found.

    Args:
        expected_status: the status that the job should be at. If None, status does not matter
        job_id_field: the name of the parameter or field that contains the job_id. Default is 'job_id'
    """

    def dependency_injector(
        redis_connection: Redis = Depends(get_redis_connection),
        job_id: str = Depends(get_job_id_dependency(job_id_field=job_id_field)),
        app_token: Optional[str] = Depends(get_bearer_token),
    ) -> auth_service.Credentials:
        """Gets a valid app_token-job_id pair with the expected job status.

        Args:
            job_id: the job_id as got from the parameters or from the uploaded file
            redis_connection: the connection to the redis database
            app_token: the app_token as got from the FastAPI request

        Raises:
            HTTPException: status_code=401, detail=job {credentials.job_id} does not exist for current user
            HTTPException: status_code=403, detail=job {credentials.job_id} is already {auth_log.status}
            InvalidJobIdInUploadedFileError: f"The job does not have a valid UUID4 {job_id_field}"
        """
        credentials = auth_service.Credentials(job_id=job_id, app_token=f"{app_token}")
        try:
            auth_service.authenticate(
                redis_connection,
                credentials=credentials,
                expected_status=expected_status,
            )
        except auth_service.AuthenticationError as exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=f"{exp}"
            )
        except auth_service.AuthorizationError as exp:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"{exp}")

        return credentials

    return dependency_injector


def get_bearer_token(
    request: Request, raise_if_error: bool = settings.IS_AUTH_ENABLED
) -> Optional[str]:
    """Extracts the bearer token from the request.

    It throws a 401 exception if not exist and `raise_if_error` is False

    Args:
        request: the request object from FastAPI
        raise_if_error: whether an error should be raised if it occurs.
            defaults to settings.IS_AUTH_ENABLED

    Raises:
        HTTPException: Unauthorized

    Returns:
        the bearer token as a string or None if it does not exist and `raise_if_error` is False
    """
    try:
        authorization_header = request.headers["Authorization"]
        return authorization_header.split("Bearer ")[1].strip()
    except (KeyError, IndexError):
        if raise_if_error:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


def get_kernel_connection() -> Connection:
    """Dependency injector for the Connection object to the kernel service"""
    _, conn = get_kernel()
    return conn


def get_kernel() -> Tuple[mp.Process, Connection]:
    """Gets kernel service's process and connection"""
    global _kernel_process, _kernel_connection

    if _kernel_connection.closed:
        # if the connection is closed, reopen
        _kernel_process, _kernel_connection = kernel_service.connect()

    return _kernel_process, _kernel_connection
