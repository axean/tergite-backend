# This code is part of Tergite
#
# (C) Axel Andersson (2022)
# (C) Chalmers Next Labs (2025)
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
from typing import Dict, List, Optional, Type

from qiskit.qobj import PulseQobjConfig, PulseQobjExperiment, PulseQobjInstruction
from quantify_scheduler import Schedule
from quantify_scheduler.compilation import determine_absolute_timing
from quantify_scheduler.resources import ClockResource

from app.libs.quantum_executor.base.experiment import (
    NativeExperiment,
    copy_expt_header_with,
)
from app.libs.quantum_executor.quantify.channel import QuantifyChannel

from ..base.quantum_job.dtos import NativeQobjConfig
from .channel import QuantifyChannelRegistry
from .instruction import (
    QBLOX_TIMEGRID_INTERVAL,
    AcquireInstruction,
    BaseInstruction,
    DelayInstruction,
    InitialObjectInstruction,
    ParamPulseInstruction,
    PulseLibInstruction,
    SetFreqInstruction,
    SetPhaseInstruction,
    ShiftFreqInstruction,
    ShiftPhaseInstruction,
)

# Map name => BaseInstruction
_INSTRUCTION_MAP: Dict[str, Type[BaseInstruction]] = {
    "setf": SetFreqInstruction,
    "shiftf": ShiftFreqInstruction,
    "setp": SetPhaseInstruction,
    "fc": ShiftPhaseInstruction,
    "delay": DelayInstruction,
    "acquire": AcquireInstruction,
    "parametric_pulse": ParamPulseInstruction,
}


@dataclass(frozen=True)
class QuantifyExperiment(NativeExperiment):
    channel_registry: QuantifyChannelRegistry
    buffer_time: float = 0.0

    # the interval between grid lines in the time grid used by Q1ASM
    timegrid_interval: float = QBLOX_TIMEGRID_INTERVAL

    @property
    def schedule(self: "QuantifyExperiment") -> Schedule:
        raw_schedule = Schedule(name=self.header.name, repetitions=self.config.shots)

        root_instruction = InitialObjectInstruction()
        raw_schedule.add(
            ref_op=None,
            ref_pt="end",
            ref_pt_new="start",
            rel_time=0.0,
            label=root_instruction.label,
            operation=root_instruction.to_operation(config=self.config),
        )

        for channel in self.channel_registry.values():  # type: QuantifyChannel
            if (
                len(channel.instructions) == 1
                and channel.instructions[0].name == "delay"
            ):
                # if the channel contains a single instruction and that instruction is a delay,
                # then do not schedule any operations on that channel
                print("\nNO DELAY\n")
                continue

            prev = root_instruction
            for curr in channel.instructions:
                rel_time = curr.t0 - prev.final_timestamp + self.timegrid_interval
                ref_op = prev.label

                raw_schedule.add(
                    ref_op=ref_op,
                    ref_pt="end",
                    ref_pt_new="start",
                    rel_time=rel_time,
                    label=curr.label,
                    operation=curr.to_operation(config=self.config),
                )

                # set the previous to the current
                prev = curr

        return _get_absolute_timed_schedule(
            schedule=raw_schedule, channel_registry=self.channel_registry
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
        channel_registry = QuantifyChannelRegistry()

        for inst in expt.instructions:
            _add_instruction_to_channel_registry(
                channel_registry=channel_registry,
                qobj_inst=inst,
                config=qobj_config,
                native_config=native_config,
                hardware_map=hardware_map,
            )

        return cls(
            header=header,
            config=qobj_config,
            channel_registry=channel_registry,
        )


def _add_instruction_to_channel_registry(
    channel_registry: QuantifyChannelRegistry,
    qobj_inst: PulseQobjInstruction,
    config: PulseQobjConfig,
    native_config: NativeQobjConfig,
    hardware_map: Dict[str, str] = None,
):
    """Extracts PulseQobjInstruction and attaches the extracted native instructions to channel_registry

    Args:
        channel_registry: the registry of all the channels to which instructions are to be attached
        qobj_inst: the PulseQobjInstruction from which instructions are to be extracted
        config: config of the pulse qobject
        native_config: the native config for the qobj
        hardware_map: the map describing the layout of the quantum device
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

    for instruction in cls.list_from_qobj_inst(
        qobj_inst,
        config=config,
        native_config=native_config,
        hardware_map=hardware_map,
        channel_registry=channel_registry,
    ):
        instruction.register()


def _get_absolute_timed_schedule(
    schedule: Schedule, channel_registry: QuantifyChannelRegistry
) -> Schedule:
    """Returns a new schedule with absolute timing

    Args:
        schedule: the raw schedule to compile
        channel_registry: the iterable of QuantifyChannel's to which are attached ClockResource's

    Returns:
        the schedule with absolute time for each operation has been
        determined.
    """
    for channel in channel_registry.values():
        clock = ClockResource(name=channel.clock, freq=channel.final_frequency)
        schedule.add_resource(clock)

    return determine_absolute_timing(schedule)
