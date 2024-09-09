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

import numpy as np
import xarray
from qiskit.qobj import PulseQobj

import app.libs.storage_file as storagefile
from app.libs.quantum_executor.base.executor import QuantumExecutor
from app.libs.quantum_executor.base.experiment import BaseExperiment
from app.libs.quantum_executor.qiskit.experiment import QiskitDynamicsExperiment
from app.libs.quantum_executor.utils.channel import Channel
from app.libs.quantum_executor.utils.instruction import Instruction
from .backend import FakeOpenPulse1Q
from .transpile import transpile


class QiskitDynamicsExecutor(QuantumExecutor):
    def __init__(self, config_file):
        super().__init__()

        self.backend = FakeOpenPulse1Q()

    def run(self, experiment: BaseExperiment, /) -> xarray.Dataset:
        job = self.backend.run(experiment.schedule)
        result = job.result()
        return result.data()["memory"]

    def construct_experiments(self, qobj: PulseQobj, /):
        # storage array
        tx = list()
        for experiment_index, experiment in enumerate(qobj.experiments):
            # TODO SIM: This whole thing to translate everything into a wrapper object for the instruction is not necessary
            # - We can directly take the qobj as it is
            # - In the qobj there is a list of experiments
            # - We have to iterate over the experiments and create a schedule for each of them

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
            # TODO SIM: If we wanted to get rid of this overly complicated notation of the Experiment object, we would have to check where it is used in the storage file as well
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

    def close(self):
        pass


class QiskitDynamicsPulseSimulator1Q(QuantumExecutor):
    def __init__(self, config_file):
        super().__init__()
        # TODO: Use measurement level provided by the client request if discriminator is not provided
        self.backend = FakeOpenPulse1Q(meas_level=1, meas_return="single")
        self.shots = 1024

    def run(self, experiment: BaseExperiment, /) -> xarray.Dataset:
        job = self.backend.run(experiment, shots=self.shots)
        result = job.result()
        data = result.data()["memory"]

        # Combine real and imaginary parts into complex numbers
        complex_data = data[:, 0, 0] + 1j * data[:, 0, 1]
        # Create acquisition index coordinate that matches the length of complex_data
        acq_index = np.arange(
            complex_data.shape[0]
        )  # Should match the number of rows in complex_data

        coords = {
            "acq_index_0": acq_index,  # Coordinate array that matches the dimension length
        }

        # Create the xarray Dataset
        ds = xarray.Dataset(
            data_vars={"0": (["acq_index_0"], complex_data)}, coords=coords
        )
        return ds

    def construct_experiments(self, qobj: PulseQobj, /):
        # because we avoid experiments structure we have to pass shots and measurement level configurations to the run function
        self.shots = qobj.config.shots
        qobj_dict = qobj.to_dict()
        tx = transpile(qobj_dict)

        self.logger.info(f"Translated {len(tx)} OpenPulse experiments.")
        return tx

    def close(self):
        pass
