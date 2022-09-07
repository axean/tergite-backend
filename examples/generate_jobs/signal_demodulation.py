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

from argparse import ArgumentParser
import json
import pathlib
import settings
from uuid import uuid4

import numpy as np
import requests

# settings
BCC_MACHINE_ROOT_URL = settings.BCC_MACHINE_ROOT_URL

REST_API_MAP = {"jobs": "/jobs"}


def gen_array(option):

    fn_dispatcher = {
        "linspace": np.linspace,
        "geomspace": np.geomspace,
        "logspace": np.logspace,
    }

    if option[0] == "stepspace":
        return ("start=" + option[1], "stop=" + option[2], "step=" + option[3])
    else:
        numpy_array = fn_dispatcher[option[0]](
            *(list(map(float, option[1:3])) + [int(option[3])])
        )
        return numpy_array.tolist()  # json ready


def main():

    description = "generate parameters for demodulation job"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-s",
        "--signal",
        action="store",
        dest="signal",
        type=str,
        nargs="*",
        default=["linspace", "0", "10", "5"],
        metavar="PARAM",
        help="Examples: -s linspace 0 10 51",
    )
    parser.add_argument(
        "-d",
        "--demod",
        action="store",
        dest="demod",
        type=str,
        nargs="*",
        default=["geomspace", "1", "9", "4"],
        help="Examples: -d geomspace 1 15 4",
    )
    opts = parser.parse_args()

    print("Signal arguments: {}".format(opts.signal))
    print("Demod arguments: {}".format(opts.demod))

    signal_array = gen_array(opts.signal)
    demod_array = gen_array(opts.demod)

    job = {
        "job_id": str(uuid4()),
        "type": "script",
        "name": "demodulation_scenario",
        "params": {
            "Sine - Frequency": signal_array,
            "Demod - Modulation frequency": demod_array,
        },
    }

    file = pathlib.Path("/tmp") / str(uuid4())
    with file.open("w") as dest:
        json.dump(job, dest)

    with file.open("r") as src:
        files = {"upload_file": src}
        url = str(BCC_MACHINE_ROOT_URL) + REST_API_MAP["jobs"]
        response = requests.post(url, files=files)

        if response:
            print("Job has been successfully sent")

    file.unlink()


if __name__ == "__main__":
    main()
