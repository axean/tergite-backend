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
from fastapi.requests import Request

from .exc import IpNotAllowedError


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
