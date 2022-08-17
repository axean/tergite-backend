# This code is part of Tergite
#
# (C) Copyright David Wahlstedt 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import numpy as np
from uuid import uuid4


# The following two functions only serve as placeholders for real
# measurement jobs to be implemented in coming versions of this code.

def mk_job_check_sig_demod():
    # here we should do something simpler than in the calibration fn
    signal_array = gen_array(["linspace", "0", "5", "5"])
    demod_array = gen_array(["geomspace", "1", "5", "4"])

    job = {
        "job_id": str(uuid4()),
        "type": "script",
        "is_calibration_sup_job": True, # job requested by calibration framework
        "name": "demodulation_scenario",
        "params": {
            "Sine - Frequency": signal_array,
            "Demod - Modulation frequency": demod_array,
        },
    }

    return job


def mk_job_calibrate_sig_demod():
    signal_array = gen_array(["linspace", "0", "10", "5"])
    demod_array = gen_array(["geomspace", "1", "9", "4"])

    job = {
        "job_id": str(uuid4()),
        "type": "script",
        "is_calibration_sup_job": True,
        "name": "demodulation_scenario",
        "params": {
            "Sine - Frequency": signal_array,
            "Demod - Modulation frequency": demod_array,
        },
    }
    return job


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
        return numpy_array.tolist()
