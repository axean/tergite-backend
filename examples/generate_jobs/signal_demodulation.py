from argparse import ArgumentParser
import numpy as np
import json
from uuid import uuid4
import requests
import pathlib


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
            *(list(map(float, option[1:2])) + [int(option[3])])
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
        default=["linspace", "0", "10", "51"],
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
        default=["stepspace", "1", "9", "4"],
        help="Examples: -d stepspace 1 9 4",
    )
    opts = parser.parse_args()

    print("Signal arguments: {}".format(opts.signal))
    print("Demod arguments: {}".format(opts.demod))

    signal_array = gen_array(opts.signal)
    demod_array = gen_array(opts.demod)

    job = {
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
        url = "http://qtl-bcc-1.qdp.chalmers.se:5000/jobs"
        response = requests.post(url, files=files)

        if response:
            print("Job has been successfully sent")

    file.unlink()


if __name__ == "__main__":
    main()
