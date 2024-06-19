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

from functools import partial

import xarray
from qiskit.qobj import PulseQobj

import app.libs.storage_file as storagefile
from app.libs.quantum_executor.channel import Channel
from app.libs.quantum_executor.executor_base import QuantumExecutor
from app.libs.quantum_executor.experiment_base import BaseExperiment
from app.libs.quantum_executor.instruction import Instruction
from app.libs.quantum_executor.qiskit.experiment import QiskitDynamicsExperiment


class QiskitDynamicsExecutor(QuantumExecutor):
    def construct_experiments(self, qobj: PulseQobj, /):
        # TODO: This function could be refactored and put on the QuantumExecutor level as static helper function
        # -> The only call on self is for the logger
        # -> Logger could be global variable anyways
        # storage array
        tx = list()

        for experiment_index, experiment in enumerate(qobj.experiments):
            instructions = map(
                partial(Instruction.from_qobj, config=qobj.config),
                experiment.instructions,
            )
            instructions = [item for sublist in instructions for item in sublist]

            # create a nice name for the experiment.
            experiment.header.name = storagefile.StorageFile.sanitized_name(
                experiment.header.name, experiment_index + 1
            )

            # convert OpenPulse experiment to Qiskit Dynamics schedule
            tx.append(
                QiskitDynamicsExperiment(
                    header=experiment.header,
                    instructions=instructions,
                    config=qobj.config,
                    channels=frozenset(
                        Channel(
                            clock=i.channel,
                            frequency=0.0,
                        )
                        for i in instructions
                    ),
                    logger=self.logger,
                )
            )

        self.logger.info(f"Translated {len(tx)} OpenPulse experiments.")
        return tx

    def run(self, experiment: BaseExperiment, /) -> xarray.Dataset:
        # TODO: How do we run?
        pass

    def close(self):
        pass

    def __init__(self):
        super().__init__()
