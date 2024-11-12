# This code is part of Tergite
#
# (C) Stefan Hill, Pontus Vikstål (2024)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import List

from qiskit.pulse import (
    Schedule,
    SetFrequency,
    DriveChannel,
    ControlChannel,
    ShiftFrequency,
    SetPhase,
    ShiftPhase,
    Delay,
    Acquire,
    AcquireChannel,
    MemorySlot,
    Play,
)
from qiskit.pulse.library import Gaussian
import datetime
import numpy as np
from qiskit.pulse.library import Waveform
from .functions import omega_c


def transpile(qobj: dict) -> List[Schedule]:
    frequency_operation_map_ = {"setf": SetFrequency, "shiftf": ShiftFrequency}
    phase_operation_map_ = {"setp": SetPhase, "fc": ShiftPhase}

    experiments = []

    for experiment_definition in qobj["experiments"]:
        t0_: str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        schedule = Schedule(name=f"open-pulse-generated-{t0_}")

        for instruction in experiment_definition["instructions"]:
            is_measurement = instruction["name"] == "acquire"
            if not is_measurement:
                channel = int(
                    instruction["ch"].strip("d").strip("readout").strip("m").strip("u")
                )
                is_measurement: bool = str(instruction["ch"]).startswith(
                    ("readout", "m")
                )

            operation = None
            if (
                instruction["name"] in frequency_operation_map_.keys()
                and not is_measurement
            ):
                operation = frequency_operation_map_[instruction["name"]](
                    instruction["frequency"] * 1e9, DriveChannel(channel)
                )

            elif (
                instruction["name"] in phase_operation_map_.keys()
                and not is_measurement
            ):
                operation = phase_operation_map_[instruction["name"]](
                    instruction["phase"], DriveChannel(channel)
                )

            elif instruction["name"] == "delay":
                if not "parameters" in instruction:
                    operation = Delay(instruction["duration"], DriveChannel(channel))
                else:
                    operation = Delay(
                        instruction.parameters["duration"], DriveChannel(channel)
                    )

            elif (
                instruction["name"] == "parametric_pulse"
                and instruction["pulse_shape"] == "constant"
            ):
                operation = Acquire(
                    1,  # set duration to 0, because it does not matter
                    AcquireChannel(channel),
                    MemorySlot(channel),
                )

            elif (
                instruction["name"] == "parametric_pulse"
                and instruction["pulse_shape"] == "gaussian"
            ):
                # amp values is no longer complex
                # TODO: find ref to the docs update
                # TODO: check with older client version

                gauss = Gaussian(
                    int(instruction["parameters"]["duration"]),
                    instruction["parameters"]["amp"],
                    float(instruction["parameters"]["sigma"]),
                )
                operation = Play(gauss, DriveChannel(channel))
            elif (
                instruction["name"] == "parametric_pulse"
                and instruction["pulse_shape"] == "wacqt_cz_gate_pulse"
            ):
                args = instruction["parameters"]
                t_gate = (
                    args["t_p"] + args["t_rf"] + 2 * args["t_w"]
                )  # total time of gate

                # Generate the time array centered in each dt interval
                time_array = np.linspace(0, t_gate, args["duration"])

                # # Compute omega_c_array
                omega_c_array = omega_c(time_array, args) - omega_c(0, args)

                wf = Waveform(samples=omega_c_array, limit_amplitude=False)

                # TODO: pass control channel that corresponds to target, control tuple
                operation = Play(wf, ControlChannel(channel))

            if operation is not None:
                schedule = schedule.insert(int(instruction["t0"]), operation)

        experiments.append(schedule)
    return experiments
