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

import retworkx as rx
from pandas import DataFrame
from qiskit.qobj import PulseQobjConfig, QobjExperimentHeader
from quantify_scheduler import Schedule

from .schedule import SimulationSchedule
from ..utils.general import rot_left
from ..utils.logger import ExperimentLogger
from .channel import Channel
from .instruction import Instruction, initial_object
from .program import QuantifyProgram, QuTipProgram


@dataclass(frozen=True)
class Experiment(abc.ABC):
    header: QobjExperimentHeader
    instructions: List[Instruction]
    config: PulseQobjConfig
    channels: FrozenSet[Channel]
    logger: ExperimentLogger
    buffer_time: float = 0.0

    @cached_property
    def dag(self: "QuantifyExperiment"):
        dag = rx.PyDiGraph(check_cycle=True, multigraph=False)

        prev_index = dict()
        for j in sorted(self.instructions, key=lambda j: j.t0):
            if j.channel not in prev_index.keys():
                # add the first non-trivial instruction on the channel
                prev_index[j.channel] = dag.add_node(j)
            else:
                # get node index of previous instruction
                i = dag[prev_index[j.channel]]

                # add the next instruction
                prev_index[j.channel] = dag.add_child(
                    parent=prev_index[j.channel], obj=j, edge=j.t0 - (i.t0 + i.duration)
                )

        return dag

    @property
    @abc.abstractmethod
    def schedule(self):
        pass

    @property
    def timing_table(self: "Experiment") -> DataFrame:
        df = self.schedule.timing_table.data
        df.sort_values("abs_time", inplace=True)
        return df


@dataclass(frozen=True)
class QuantifyExperiment(Experiment):

    @property
    def schedule(self: "QuantifyExperiment") -> Schedule:
        self.logger.info(f"Compiling {self.header.name}")
        prog = QuantifyProgram(
            name=self.header.name,
            channels=self.channels,
            config=self.config,
            logger=self.logger,
        )
        prog.schedule_operation(initial_object, ref_op=None, rel_time=0.0)

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

                prog.schedule_operation(
                    self.dag[idx],
                    rel_time=rel_time,
                    ref_op=ref_op,
                )

        return prog.compiled_schedule


@dataclass(frozen=True)
class QuTipExperiment(Experiment):

    @property
    def schedule(self: 'QuTipExperiment') -> 'SimulationSchedule':
        # TODO: Override this completely
        self.logger.info(f"Compiling {self.header.name}")
        # TODO: Check whether it is even necessary to have this Program helper class with the simulator
        prog = QuTipProgram(
            name=self.header.name,
            channels=self.channels,
            config=self.config,
            logger=self.logger,
        )
        wccs = rx.weakly_connected_components(self.dag)

        for wcc in wccs:
            wcc_nodes = list(sorted(list(wcc)))

            prog.schedule_operation([self.dag[i_] for i_ in wcc_nodes])
        return prog.schedule
