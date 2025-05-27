# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020, 2021
# (C) Copyright Martin Ahindura 2023
# (C) Copyright Chalmers Next Labs 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Utilities to do with HTTP APIs"""
import logging
import shutil
from pathlib import Path
from typing import Awaitable, Callable, Optional, Union

import requests
from fastapi import HTTPException, Request, Response, UploadFile
from fastapi.exception_handlers import http_exception_handler

import settings


def get_mss_client(app_token: str = settings.MSS_APP_TOKEN) -> requests.Session:
    """Returns an MSS client to be used to make HTTP queries to MSS

    Args:
        app_token: the app token to use when making authenticated requests

    Returns:
        the requests.Session that can query MSS
    """
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {app_token}"})
    return session


def save_uploaded_file(file: UploadFile, target: Path) -> Path:
    """Saves the uploaded file to the given target path

    Args:
        file: the file to upload
        target: the target path to save to

    Returns:
        the new path to the saved file
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    file.file.seek(0)
    with target.open("wb") as destination:
        shutil.copyfileobj(file.file, destination)
    file.file.close()

    return target


def to_http_error(
    status_code: int, custom_message: Optional[str] = None
) -> Callable[[Request, Exception], Union[Response, Awaitable[Response]]]:
    """An error handler that converts the exception to an HTTPException

    The details in the http error are got from the exception itself.
    It also logs the original error.

    Args:
        status_code: the HTTP status code
        custom_message: a custom message to send to the client

    Returns:
        an HTTP exception handler function
    """

    async def handler(request: Request, exp: Exception) -> Response:
        logging.error(exp)
        message = custom_message
        if message is None:
            message = f"{exp}"

        http_exp = HTTPException(status_code, message)
        return await http_exception_handler(request, http_exp)

    return handler
