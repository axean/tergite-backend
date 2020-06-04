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
    scenario_template_filepath = Path("./template.json")
    calibration_filepath = Path("./calibration.json")

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
    update_step_single_value(s, "QA - Samples", qobj["config"].get("shots", 1024))

    # update with latest calibration data, if it exists
    if calibration_filepath.exists():
        with open(calibration_filepath, "r") as f:
            calibration = json.load(f)
        update_calibration_data(s, calibration)

    # set metadata
    s.log_name = "Test qobj"
    s.comment = "Comment for log"
    s.tags.project = "My project"
    s.tags.user = "Chalmers default user"
    s.tags.tags = ["Qobj"]

    # set timing info
    s.wait_between = 0.2

    return s


def update_step_single_value(scenario, name, value):
    scenario.get_step(name).range_items[0].single = value


def translate_parameter_name(id_, calibration_parameter):
    parameters = {
        "qubit_frequency": "Qubit {id} Frequency",
        "pi_amplitude": "Qubit {id} Amp",
        "drag_coefficient": "Qubit {id} Alpha",
        "readout_frequency": "pulses - Readout frequency #{id}",
        "readout_amplitude": "pulses - Readout amplitude #{id}",
    }

    return parameters[calibration_parameter].format(id=str(id_))


def update_calibration_data(scenario, calibration):
    for qubit in calibration["qubits"]:
        id_ = qubit["id"]
        for parameter in qubit:
            if parameter != "id":
                update_step_single_value(
                    scenario, translate_parameter_name(id_, parameter), qubit[parameter]
                )
