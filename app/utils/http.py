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
"""Utilities to do with HTTP"""
import requests

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
