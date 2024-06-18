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

import retworkx as rx

from ..experiment_base import BaseExperiment
from ..program.qutip import QuTipProgram
from ..schedule import SimulationSchedule


@dataclass(frozen=True)
class QuTipExperiment(BaseExperiment):
    @property
    def schedule(self: "QuTipExperiment") -> "SimulationSchedule":
        self.logger.info(f"Compiling {self.header.name}")
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
