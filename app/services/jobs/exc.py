# This code is part of Tergite
#
# (C) Nicklas Botö, Fabian Forslund 2022
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
"""Exceptions related to the jobs service"""
from app.utils.exc import BaseBccException


class JobNotFound(Exception):
    """A job was not found on redis"""

    def __init__(self, job_id) -> None:
        self.job_id = job_id

    def __str__(self):
        return f"Job {self.job_id} not found"


class MalformedJob(BaseBccException):
    """Exception when Job file is malformed"""


class JobCancelled(BaseBccException):
    """Exception when Job is cancelled"""
