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

import xarray
from qiskit.qobj import PulseQobj

from app.services.quantum_executor.scheduler.experiment import Experiment
from app.services.quantum_executor.base import QuantumExecutor


class QuTipExecutor(QuantumExecutor):

    def __init__(self):
        super().__init__()

    def construct_experiments(self, qobj: PulseQobj, /):
        pass

    def run(self, experiment: Experiment, /) -> xarray.Dataset:
        pass

    def close(self):
        pass
