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
"""Utility functions for API related code"""
from typing import Optional

from fastapi import HTTPException, status
from fastapi.requests import Request


def get_bearer_token(request: Request, raise_if_error: bool = True) -> Optional[str]:
    """Extracts the bearer token from the request or throws a 401 exception if not exist and `raise_if_error` is False

    Args:
        request: the request object from FastAPI
        raise_if_error: whether an error should be raised if it occurs

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
