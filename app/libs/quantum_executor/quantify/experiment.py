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
# Refactored by Stefan Hill (2024)
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Iterable, List, Optional, Type

import numpy as np
import qiskit.pulse
import retworkx as rx
from fontTools.ttLib.tables.ttProgram import instructions
from qiskit import schedule
from qiskit.qobj import PulseQobjConfig, PulseQobjExperiment, PulseQobjInstruction
from quantify_scheduler import Operation, Schedule
from quantify_scheduler.compilation import determine_absolute_timing
from quantify_scheduler.resources import ClockResource

from app.libs.quantum_executor.base.experiment import (
    NativeExperiment,
    copy_expt_header_with,
)
from app.libs.quantum_executor.base.instruction import Instruction
from app.libs.quantum_executor.utils.channel import Channel
from app.libs.quantum_executor.utils.general import flatten_list, rot_left

from ..base.quantum_job.dtos import NativeQobjConfig
from .instruction import (
    AcquireInstruction,
    DelayInstruction,
    FreqInstruction,
    InitialObjectInstruction,
    ParamPulseInstruction,
    PhaseInstruction,
    PulseLibInstruction,
)

# FIXME: Why is this initial object hard coded here?
initial_object = InitialObjectInstruction()

# Map name => Instruction
_INSTRUCTION_MAP: Dict[str, Type[Instruction]] = {
    "setf": FreqInstruction,
    "setp": PhaseInstruction,
    "fc": PhaseInstruction,
    "delay": DelayInstruction,
    "acquire": AcquireInstruction,
    "parametric_pulse": ParamPulseInstruction,
}


@dataclass(frozen=True)
class QuantifyExperiment(NativeExperiment):
    @property
    def schedule(self: "QuantifyExperiment") -> Schedule:
        channel_map: Dict[str, Channel] = {ch.clock: ch for ch in self.channels}
        raw_schedule = Schedule(name=self.header.name, repetitions=self.config.shots)
        _schedule_instruction(
            schedule=raw_schedule,
            instruction=initial_object,
            ref_op=None,
            rel_time=0.0,
            config=self.config,
        )

        wccs = rx.weakly_connected_components(self.dag)

        for wcc in wccs:
            wcc_nodes = list(sorted(list(wcc)))

            # if the channel contains a single instruction and that instruction is a delay,
            # then do not schedule any operations on that channel
            if len(wcc_nodes) == 1:
                if self.dag[wcc_nodes[0]].name == "delay":
                    print()
                    print("NO DELAY")
                    print()
                    continue

            # else, schedudle the instructions on the channels
            for n, idx in enumerate(wcc_nodes):
                ref_idx = next(iter(rot_left(reversed(wcc_nodes[: n + 1]), 1)))
                if ref_idx == idx:
                    ref_op = initial_object.label
                    rel_time = self.buffer_time
                else:
                    ref_op = self.dag[ref_idx].label
                    rel_time = self.dag.get_edge_data(ref_idx, idx) + 4e-9

                instruction: Instruction = self.dag[idx]
                _schedule_instruction(
                    schedule=raw_schedule,
                    instruction=instruction,
                    rel_time=rel_time,
                    ref_op=ref_op,
                    config=self.config,
                    channel=channel_map[instruction.channel],
                )

        return _get_absolute_timed_schedule(
            schedule=raw_schedule, channels=channel_map.values()
        )

    @classmethod
    def from_qobj_expt(
        cls,
        expt: PulseQobjExperiment,
        name: str,
        qobj_config: PulseQobjConfig,
        native_config: NativeQobjConfig,
        hardware_map: Optional[Dict[str, str]],
    ) -> "QuantifyExperiment":
        """Converts PulseQobjExperiment to native experiment

        Args:
            expt: the pulse qobject experiment to translate
            name: the name of the experiment
            qobj_config: the pulse qobject config
            native_config: the native config for the qobj
            hardware_map: the map of the real/simulated device to the logical definitions

        Returns:
            the QiskitDynamicsExperiment corresponding to the PulseQobj
        """
        header = copy_expt_header_with(expt.header, name=name)
        inst_nested_list = (
            _extract_instructions(
                qobj_inst=inst,
                config=qobj_config,
                native_config=native_config,
                hardware_map=hardware_map,
            )
            for inst in expt.instructions
        )
        native_instructions = flatten_list(inst_nested_list)

        return cls(
            header=header,
            instructions=native_instructions,
            config=qobj_config,
            channels=frozenset(
                Channel(
                    clock=i.channel,
                    frequency=0.0,
                )
                for i in native_instructions
            ),
        )


def _extract_instructions(
    qobj_inst: PulseQobjInstruction,
    config: PulseQobjConfig,
    native_config: NativeQobjConfig,
    hardware_map: Dict[str, str] = None,
) -> List[Instruction]:
    """Extracts tergite-specific instructions from the PulseQobjInstruction

    Args:
        qobj_inst: the PulseQobjInstruction from which instructions are to be extracted
        config: config of the pulse qobject
        native_config: the native config for the qobj
        hardware_map: the map describing the layout of the quantum device

    Returns:
        list of tergite-specific instructions
    """
    if hardware_map is None:
        hardware_map = {}

    try:
        cls = _INSTRUCTION_MAP[qobj_inst.name]
    except KeyError as exp:
        if qobj_inst.name in config.pulse_library:
            cls = PulseLibInstruction
        else:
            raise RuntimeError(
                f"No mapping for PulseQobjInstruction {qobj_inst}.\n{exp}"
            )

    return cls.list_from_qobj_inst(
        qobj_inst, config=config, native_config=native_config, hardware_map=hardware_map
    )


def _schedule_instruction(
    schedule: Schedule,
    instruction: Instruction,
    rel_time: float,
    config: PulseQobjConfig,
    ref_op: Optional[str] = None,
    channel: Optional[Channel] = None,
):
    # FIXME: Move all these if contents into the respective classes
    # -----------------------------------------------------------------
    if instruction.name in {"setp", "setf", "fc", "shiftf"}:
        # relative phase change
        if instruction.name == "fc":
            channel.phase += instruction.phase

        # absolute phase change
        elif instruction.name == "setp":
            channel.phase = instruction.phase

        # relative frequency change
        elif instruction.name == "shiftf":
            channel.frequency += instruction.frequency

        # absolute frequency change
        elif instruction.name == "setf":
            channel.frequency = instruction.frequency

        else:
            raise RuntimeError(f"Unable to execute command {instruction}.")

    if instruction.name in {
        "setp",
        "setf",
        "fc",
        "shiftf",
        "initial_object",
        "delay",
    }:
        operation = Operation(name=instruction.unique_name)
        operation.data["pulse_info"] = [
            {
                "wf_func": None,
                "t0": 0.0,
                "duration": instruction.duration,
                "clock": instruction.channel,
                "port": None,
            }
        ]
        operation._update()

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
        operation.data["acquisition_info"] = [
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
                "acq_index": channel.acquisitions,
                "bin_mode": instruction.bin_mode,
                "acq_return_type": instruction.acq_return_type,
                "protocol": instruction.protocol,
            }
        ]

        operation._update()
        channel.acquisitions += 1  # increment no. of acquisitions

    elif instruction.name == "parametric_pulse":
        wf_fn = getattr(
            qiskit.pulse.library.discrete, str.lower(instruction.pulse_shape)
        )
        waveform = wf_fn(**instruction.parameters).samples
        operation = _generate_numerical_pulse(
            channel=channel, instruction=instruction, waveform=waveform
        )

    elif instruction.name in config.pulse_library:
        waveform = config.pulse_library[instruction.name]
        operation = _generate_numerical_pulse(
            channel=channel, instruction=instruction, waveform=waveform
        )

    else:
        raise RuntimeError(f"Unable to schedule operation {instruction}.")

    schedule.add(
        ref_op=ref_op,
        ref_pt="end",
        ref_pt_new="start",
        rel_time=rel_time,
        label=instruction.label,
        operation=operation,
    )


def _get_absolute_timed_schedule(
    schedule: Schedule, channels: Iterable[Channel]
) -> Schedule:
    """Returns a new schedule with absolute timing

    Args:
        schedule: the raw schedule to compile
        channels: the iterable of Channel's to which are attached ClockResource's

    Returns:
        the schedule with absolute time for each operation has been
        determined.
    """
    for channel in channels:
        clock = ClockResource(name=channel.clock, freq=channel.frequency)
        schedule.add_resource(clock)

    return determine_absolute_timing(schedule)


def _generate_numerical_pulse(
    channel: Channel, instruction: Instruction, waveform: np.ndarray
) -> Operation:
    """Generates a numerical pulse on the given channel for the given instruction given a particular waveform

    Args:
        channel: the channel on which the pulse is to be sent
        instruction: the raw instruction
        waveform: the points that form the samples from which the numerical pulse is to be generated

    Returns:
        Operation representing the numerical pulse
    """
    waveform *= np.exp(1.0j * channel.phase)
    operation = Operation(name=instruction.unique_name)
    operation.data["pulse_info"] = [
        {
            "wf_func": "quantify_scheduler.waveforms.interpolated_complex_waveform",
            "samples": waveform.tolist(),
            "t_samples": np.linspace(0, instruction.duration, len(waveform)).tolist(),
            "duration": instruction.duration,
            "interpolation": "linear",
            "clock": instruction.channel,
            "port": instruction.port,
            "t0": 0.0,
        }
    ]
    operation._update()
    return operation
