# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import pathlib
import json
import time


def execute_job(file):
    print(f"Executing file {str(file)}")

    job_dict = {}
    with file.open() as f:
        job_dict = json.load(f)

    print(f"Command: {job_dict['name']}")
    time.sleep(3)

    file.unlink()
