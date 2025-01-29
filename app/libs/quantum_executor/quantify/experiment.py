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
from typing import Dict, List, Optional, Type

import retworkx as rx
from qiskit.qobj import PulseQobjConfig, PulseQobjExperiment, PulseQobjInstruction
from quantify_scheduler import Schedule
from quantify_scheduler.compilation import determine_absolute_timing
from quantify_scheduler.resources import ClockResource

from app.libs.quantum_executor.base.experiment import (
    NativeExperiment,
    copy_expt_header_with,
)
from app.libs.quantum_executor.utils.general import flatten_list, rot_left

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

        wccs = rx.weakly_connected_components(self.dag)

        for wcc in wccs:
            wcc_nodes = sorted(list(wcc))

            # if the channel contains a single instruction and that instruction is a delay,
            # then do not schedule any operations on that channel
            if len(wcc_nodes) == 1:
                if self.dag[wcc_nodes[0]].name == "delay":
                    print("\nNO DELAY\n")
                    continue

            # else, schedule the instructions on the channels
            for n, idx in enumerate(wcc_nodes):
                ref_idx = next(iter(rot_left(reversed(wcc_nodes[: n + 1]), 1)))
                if ref_idx == idx:
                    ref_op = root_instruction.label
                    rel_time = self.buffer_time
                else:
                    ref_op = self.dag[ref_idx].label
                    rel_time = (
                        self.dag.get_edge_data(ref_idx, idx) + self.timegrid_interval
                    )

                instruction: BaseInstruction = self.dag[idx]
                raw_schedule.add(
                    ref_op=ref_op,
                    ref_pt="end",
                    ref_pt_new="start",
                    rel_time=rel_time,
                    label=instruction.label,
                    operation=instruction.to_operation(config=self.config),
                )

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

        inst_nested_list = (
            _extract_instructions(
                qobj_inst=inst,
                config=qobj_config,
                native_config=native_config,
                channel_registry=channel_registry,
                hardware_map=hardware_map,
            )
            for inst in expt.instructions
        )
        native_instructions = flatten_list(inst_nested_list)

        return cls(
            header=header,
            instructions=native_instructions,
            config=qobj_config,
            channel_registry=channel_registry,
        )


def _extract_instructions(
    qobj_inst: PulseQobjInstruction,
    config: PulseQobjConfig,
    native_config: NativeQobjConfig,
    channel_registry: QuantifyChannelRegistry,
    hardware_map: Dict[str, str] = None,
) -> List[BaseInstruction]:
    """Extracts tergite-specific instructions from the PulseQobjInstruction

    Args:
        qobj_inst: the PulseQobjInstruction from which instructions are to be extracted
        config: config of the pulse qobject
        native_config: the native config for the qobj
        channel_registry: the registry of all the channels for the given experiment
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
        qobj_inst,
        config=config,
        native_config=native_config,
        hardware_map=hardware_map,
        channel_registry=channel_registry,
    )


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
