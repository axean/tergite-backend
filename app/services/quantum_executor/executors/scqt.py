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

from .quantify import QuantifyExecutor
from ..scheduler.experiment.quantify import QuantifyExperiment
from ..simulator import scqt


class SCQTQuantifyExecutor(QuantifyExecutor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulator = scqt.Simulator()

    def run(self, experiment: QuantifyExperiment, /):
        schedule = experiment.schedule
        compiled_sched = self.simulator.compile(schedule)
        self.logger.log_schedule(compiled_sched)
        return self.simulator.run(compiled_sched, output="voltage_single_shot")
