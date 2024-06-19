# This code is part of Tergite
#
# (C) Stefan Hill (2024)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from dataclasses import dataclass
from functools import cached_property
from typing import List

import numpy as np
import qiskit.pulse

from app.libs.quantum_executor.channel import Channel
from app.libs.quantum_executor.instruction import Instruction
from app.libs.quantum_executor.program_base import BaseProgram
from app.libs.quantum_executor.qutip.schedule import (
    SimulationSchedule,
    MeasurementOperation,
    UnitaryOperation,
)


@dataclass(frozen=True)
class QiskitDynamicsProgram(BaseProgram):
    @cached_property
    def schedule(self):
        # TODO: return type qiskit schedule
        return SimulationSchedule(name=self.name)

    @cached_property
    def compiled_schedule(self):
        # TODO: return type qiskit schedule
        return self.schedule

    def get_channel(self, instruction: Instruction, /) -> "Channel":
        return next(filter(lambda ch: instruction.channel == ch.clock, self.channels))

    def update_frame(self, instruction: Instruction, /):
        # TODO: How are we doing with phase in QiskitDynamics?
        # relative phase change
        if instruction.name == "fc":
            self.get_channel(instruction).phase += instruction.phase

        # absolute phase change
        elif instruction.name == "setp":
            self.get_channel(instruction).phase = instruction.phase

        # relative frequency change
        elif instruction.name == "shiftf":
            self.get_channel(instruction).frequency += instruction.frequency

        # absolute frequency change
        elif instruction.name == "setf":
            self.get_channel(instruction).frequency = instruction.frequency

        else:
            raise RuntimeError(f"Unable to execute command {instruction}.")

    def schedule_operation(
        self,
        instructions: List["Instruction"],
        /,
    ):
        # TODO: This is to be adjusted and might be more aligned with the QuantifyProgram

        # Get the names of all instructions, so, we have a basis to do a decision which operation it is
        instruction_map = {idx_: i_.name for idx_, i_ in enumerate(instructions)}

        # FIXME: This sort of parsing is potentially unsafe
        channel = int(instructions[0].channel[1:])

        # This is to make sure that we always get all instructions inside the mapping
        assert len(instruction_map) == len(instructions)

        operation = None

        # This is to check whether the operation should be a measurement
        if "acquire" in instruction_map.values():
            t0 = 0.0

            # We determine the start time of the measurement to properly schedule it
            for i_ in instructions:
                if i_.name == "parametric_pulse":
                    t0 = i_.t0

            # We are adding a measurement to the qubit (channel) on time t0
            operation = MeasurementOperation(channel, t0=t0)

        # TODO: We currently cannot handle delays
        # In the simulator, we currently do not handle delays, because it is not necessary for the simple type
        # of operations we are having right now. If one would want to have delays in the code, one would create
        # a new type of 'Operation' for the 'SimulationSchedule'
        elif "delay" in instruction_map.values():
            pass

        # This is handling all other sort of pulses
        elif "parametric_pulse" in instruction_map.values():
            # Initialise the parameters we need to define the array that describes the pulse
            frequency = 0.0
            phase = 0.0
            amp = 0.0
            sigma = 1
            t0 = 0.0
            duration = 0.0
            pulse_shape = "gaussian"
            parameters = None
            discrete_steps = 0

            # Instructions is a list that contains all information to build an instruction
            # We have to iterate over the elements to fetch the parameters from different objects
            for i_ in instructions:
                # Some instructions contain a change in frequency
                if i_.name in ["setf", "shiftf"]:
                    frequency = i_.frequency
                # Some instructions contain a phase shift
                elif i_.name in ["setp", "fc"]:
                    phase = i_.phase
                # The parametric pulse stores all information about the shape and pulse parameters
                elif i_.name == "parametric_pulse":
                    amp = i_.parameters["amp"]
                    sigma = i_.parameters["sigma"]
                    t0 = i_.t0
                    duration = i_.duration
                    pulse_shape = i_.pulse_shape
                    parameters = i_.parameters
                    # We need the steps for the simulation model
                    discrete_steps = int(duration * 10e8)
            wf_fn = getattr(qiskit.pulse.library.discrete, str.lower(pulse_shape))
            coeffs = wf_fn(**parameters).samples
            coeffs *= np.exp(1.0j * phase).tolist()
            tlist = np.linspace(0, duration, len(coeffs)).tolist()

            operation = UnitaryOperation(
                channel,
                t0=t0,
                frequency=frequency,
                phase=phase,
                amp=amp,
                sigma=sigma,
                discrete_steps=discrete_steps,
            )

            operation.data.update(
                {
                    "pulse_info": [
                        {
                            "wf_func": "quantify_scheduler.waveforms.interpolated_complex_waveform",
                            "samples": coeffs,
                            "t_samples": tlist,
                            "duration": duration,
                            "interpolation": "linear",
                            "t0": 0.0,
                        }
                    ],
                }
            )

        if operation is not None:
            self.schedule.add(operation)
