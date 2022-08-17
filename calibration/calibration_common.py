# This code is part of Tergite
#
# (C) Johan Blomberg, Gustav Grännsjö 2020
# (C) Copyright David Wahlstedt 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import asyncio
from enum import Enum


# Used by check_data to indicate outcome
class DataStatus(Enum):
    in_spec = 1
    out_of_spec = 2
    bad_data = 3

# The event will be awaited by calibration.calibration_lib.request_job.
# Then, in calibration_supervisor.handle_message, when the incoming
# message matches self.requested_job_id, the event will be set. This
# causes request_job to resume after waiting, and it can
# clear the event.
class JobDoneEvent:
    def __init__(self, event: asyncio.Event):
        self.event = event
        self.requested_job_id = None

