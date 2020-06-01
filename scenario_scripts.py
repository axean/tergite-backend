# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek, Andreas Bengtsson 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from Labber import Scenario
import numpy as np
import json
from pathlib import Path


def demodulation_scenario(signal_array, demod_array):
    # create and add instruments
    s = Scenario()
    instr_signal = s.add_instrument("Simple Signal Generator", name="Sine")
    instr_demod = s.add_instrument("Signal Demodulation", name="Demod")

    # set a few instrument settings
    instr_demod.values["Use phase reference signal"] = False
    instr_demod.values["Length"] = 1.0

    # add signal connections between channels
    s.add_connection("Sine - Signal", "Demod - Input data")

    # add step items, values can be defined with np array or keywords
    s.add_step("Sine - Frequency", signal_array)
    s.add_step("Demod - Modulation frequency", demod_array)

    # add log channels
    s.add_log("Demod - Value")

    # set metadata
    s.comment = "Comment for log"
    s.tags.project = "My project"
    s.tags.user = "John Doe"
    s.tags.tags = ["Tag 1", "Tag 2/Subtag"]

    # set timing info
    s.wait_between = 0.01

    return s


def qobj_scenario(qobj):
    supported_gates = set(
        ["u1", "u2", "u3", "rx", "ry", "rz", "x", "y", "z", "h", "cz", "cx"]
    )
    scenario_template_filepath = Path("./qobj_template.labber")

    def validate_gate(gate):
        if gate["name"] not in supported_gates:
            raise ValueError("{} is not supported by the hardware".format(gate["name"]))

        if "conditional" in gate:
            raise ValueError("Conditional gates are not supported by the hardware")

    def validate_job(qobj):
        if qobj["type"] != "QASM":
            raise ValueError("Only QASM-type jobs are supported.")

        if len(qobj["experiments"]) > 1:
            raise ValueError("Only supports single experiments at this moment")

    validate_job(qobj)
    for ins in qobj["experiments"][0]["instructions"]:
        validate_gate(ins)

    s = Scenario(scenario_template_filepath)

    mqpg = s.get_instrument(name="pulses")
    mqpg.values["Sequence"] = "QObj"
    mqpg.values["QObj JSON"] = json.dumps(qobj["experiments"][0])

    # configure number of shots
    samples_step = s.get_step("QA - Samples")
    samples_step.range_items[0].single = 10.0

    # set metadata
    s.log_name = "Test qobj"
    s.comment = "Comment for log"
    s.tags.project = "My project"
    s.tags.user = "Chalmers default user"
    s.tags.tags = ["Qobj"]

    # set timing info
    s.wait_between = 0.2

    return s
