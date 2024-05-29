# This code is part of Tergite
#
# (C) Axel Andersson (2022)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Refactored by Martin Ahindura (2024)
import abc
from dataclasses import dataclass
from functools import cached_property
from typing import FrozenSet, List

import numpy as np
import qiskit.pulse.library.discrete
from qiskit.qobj import PulseQobjConfig
from quantify_scheduler import Schedule
from quantify_scheduler.compilation import determine_absolute_timing
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.resources import ClockResource

from .schedule import SimulationSchedule, MeasurementOperation, UnitaryOperation
from ..utils.logger import ExperimentLogger
from .channel import Channel
from .instruction import Instruction


@dataclass(frozen=True)
class Program(abc.ABC):
    name: str
    channels: FrozenSet[Channel]
    config: PulseQobjConfig
    logger: ExperimentLogger


@dataclass(frozen=True)
class QuantifyProgram(Program):

    @cached_property
    def schedule(self: "QuantifyProgram") -> "Schedule":
        return Schedule(name=self.name, repetitions=self.config.shots)

    @cached_property
    def compiled_schedule(self: "QuantifyProgram") -> "Schedule":
        for channel in self.channels:
            clock = ClockResource(name=channel.clock, freq=channel.frequency)
            self.schedule.add_resource(clock)
            self.logger.info(f"Added resource: {clock}")

        return determine_absolute_timing(self.schedule)

    def get_channel(self: "QuantifyProgram", instruction: Instruction, /) -> "Channel":
        return next(filter(lambda ch: instruction.channel == ch.clock, self.channels))

    def numerical_pulse(
            self: "QuantifyProgram", instruction: Instruction, /, *, waveform: np.ndarray
    ) -> "Operation":
        waveform *= np.exp(1.0j * self.get_channel(instruction).phase)
        operation = Operation(name=instruction.unique_name)
        operation.data.update(
            {
                "pulse_info": [
                    {
                        "wf_func": "quantify_scheduler.waveforms.interpolated_complex_waveform",
                        "samples": waveform.tolist(),
                        "t_samples": np.linspace(
                            0, instruction.duration, len(waveform)
                        ).tolist(),
                        "duration": instruction.duration,
                        "interpolation": "linear",
                        "clock": instruction.channel,
                        "port": instruction.port,
                        "t0": 0.0,
                    }
                ],
            }
        )
        operation._update()
        return operation

    def update_frame(self: "QuantifyProgram", instruction: Instruction, /):
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
            self: "QuantifyProgram", instruction: Instruction, /, *, rel_time: float, ref_op: str
    ):
        # -----------------------------------------------------------------
        if instruction.name in {"setp", "setf", "fc", "shiftf"}:
            self.update_frame(instruction)
        # -----------------------------------------------------------------

        # -----------------------------------------------------------------
        if instruction.name in {
            "setp",
            "setf",
            "fc",
            "shiftf",
            "initial_object",
            "delay",
        }:
            operation = Operation(name=instruction.unique_name)
            operation.data.update(
                {
                    "pulse_info": [
                        {
                            "wf_func": None,
                            "t0": 0.0,
                            "duration": instruction.duration,
                            "clock": instruction.channel,
                            "port": None,
                        }
                    ]
                }
            )
            operation._update()
        # -----------------------------------------------------------------
        elif instruction.name == "acquire":
            if instruction.protocol == "SSBIntegrationComplex":
                waveform_i = {
                    "port": instruction.port,
                    "clock": instruction.channel,
                    "t0": 0.0,
                    "duration": instruction.duration,
                    "wf_func": "quantify_scheduler.waveforms.square",
                    "amp": 1,
                }
                waveform_q = {
                    "port": instruction.port,
                    "clock": instruction.channel,
                    "t0": 0.0,
                    "duration": instruction.duration,
                    "wf_func": "quantify_scheduler.waveforms.square",
                    "amp": 1j,
                }
                weights = [waveform_i, waveform_q]
            elif instruction.protocol == "trace":
                weights = []

            else:
                raise RuntimeError(
                    f"Cannot schedule acquisition with unknown protocol {instruction.protocol}."
                )

            operation = Operation(name=instruction.unique_name)
            operation.data.update(
                {
                    "acquisition_info": [
                        {
                            "waveforms": weights,
                            "t0": 0.0,
                            "clock": instruction.channel,
                            "port": instruction.port,
                            "duration": instruction.duration,
                            "phase": 0.0,
                            # "acq_channel": instruction.memory_slot, # TODO: Fix deranged memory slot readout
                            "acq_channel": int(
                                instruction.channel[1:]
                            ),  # FIXME, hardcoded single character parsing
                            "acq_index": self.get_channel(instruction).acquisitions,
                            "bin_mode": instruction.bin_mode,
                            "acq_return_type": instruction.acq_return_type,
                            "protocol": instruction.protocol,
                        }
                    ]
                }
            )

            operation._update()
            self.get_channel(
                instruction
            ).acquisitions += 1  # increment no. of acquisitions
        # -----------------------------------------------------------------
        elif instruction.name == "parametric_pulse":
            wf_fn = getattr(
                qiskit.pulse.library.discrete, str.lower(instruction.pulse_shape)
            )
            waveform = wf_fn(**instruction.parameters).samples
            operation = self.numerical_pulse(instruction, waveform=waveform)
        # -----------------------------------------------------------------
        elif instruction.name in self.config.pulse_library:
            waveform = self.config.pulse_library[instruction.name]
            operation = self.numerical_pulse(instruction, waveform=waveform)
        # -----------------------------------------------------------------
        else:
            raise RuntimeError(f"Unable to schedule operation {instruction}.")
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        self.schedule.add(
            ref_op=ref_op,
            ref_pt="end",
            ref_pt_new="start",
            rel_time=rel_time,
            label=instruction.label,
            operation=operation,
        )


@dataclass(frozen=True)
class QuTipProgram(Program):

    @cached_property
    def schedule(self) -> 'SimulationSchedule':
        return SimulationSchedule(name=self.name)

    @cached_property
    def compiled_schedule(self) -> 'SimulationSchedule':
        return self.schedule

    def get_channel(self, instruction: Instruction, /) -> 'Channel':
        return next(filter(lambda ch: instruction.channel == ch.clock, self.channels))

    def update_frame(self, instruction: Instruction, /):

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
            self, instructions: List['Instruction'], /,
    ):
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
            operation = MeasurementOperation(channel,
                                             t0=t0)

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

            operation = UnitaryOperation(channel,
                                         t0=t0,
                                         frequency=frequency,
                                         phase=phase,
                                         amp=amp,
                                         sigma=sigma,
                                         discrete_steps=discrete_steps)

            operation.data.update({
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
            })

        if operation is not None:
            self.schedule.add(operation)
