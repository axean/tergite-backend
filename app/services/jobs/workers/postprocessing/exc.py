# This code is part of Tergite
#
# (C) Martin Ahindura 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Exceptions for postprocessing"""


class PostProcessingError(Exception):
    """Exception raised when something unexpected happens during postprocessing"""

    def __init__(self, exp: Exception, job_id: str):
        self.exp = exp
        self.job_id = job_id

    def __repr__(self):
        return f"{self.__class__.__name__}<job_id: {self.job_id}, {repr(self.exp)}>"
