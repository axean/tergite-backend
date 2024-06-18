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

import math
import os
from datetime import datetime
from functools import partial
from typing import Union, List

import numpy as np
import qutip as qt
import xarray
import yaml
from qiskit.qobj import PulseQobj
from quantify_core.data import handling as dh
from xarray import Dataset

from chalmers_qubit.base.operations import project_on_qubit
from chalmers_qubit.sarimner.model import SarimnerModel
from chalmers_qubit.sarimner.processor import SarimnerProcessor

import settings
from app.libs.quantum_executor.channel import Channel
from app.libs.quantum_executor.executor_base import QuantumExecutor
from app.libs.quantum_executor.instruction import Instruction
from app.libs.quantum_executor.qutip.schedule import (
    SimulationSchedule,
    UnitaryOperation,
    MeasurementOperation,
)
from app.libs.quantum_executor.qutip.experiment import QuTipExperiment
from app.libs.storage_file import file as storagefile


class QuTipExecutor(QuantumExecutor):
    def __init__(self: "QuTipExecutor", config_file: Union[str, bytes, os.PathLike]):
        super().__init__()
        dh.set_datadir(settings.EXECUTOR_DATA_DIR)
        self.backend_config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
        self.busy = False

        qubit_frequencies: List[float] = [
            q["frequency"] * 1e-9 * 2 * math.pi
            for q in self.backend_config["device_properties"]["qubit"]
        ]
        anharmonicities: List[float] = [
            0.3 * 2 * math.pi * 10e-3 for _ in range(len(qubit_frequencies))
        ]

        self.model = SarimnerModel(qubit_frequencies, anharmonicities)
        self.processor = SarimnerProcessor(self.model, compiler="None")

    def construct_experiments(self, qobj: PulseQobj, /):
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

            # convert OpenPulse experiment to Quantify schedule
            tx.append(
                QuTipExperiment(
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

    def run(self, experiment: QuTipExperiment, /) -> xarray.Dataset:
        initial_state = qt.tensor([qt.basis(3, 0)] * self.processor.num_qubits)

        # We need to initialise with time 0 for qutip to work
        measurements = [(None, 0)]

        # Initialise with 0 if nothing is happening
        schedule: "SimulationSchedule" = experiment.schedule

        qubits = list(set(map(lambda x: x.channel, schedule.operations)))
        coeffs_dict = {
            f"x{qubit}": np.zeros(schedule.discrete_steps) for qubit in qubits
        }
        tlist_dict = {
            f"x{qubit}": np.arange(schedule.discrete_steps) for qubit in qubits
        }

        # This will stepwise fill the operations into the empty pulse definitions
        for operation in schedule.operations:
            # Fill the gaps with pulses
            # If there is a measurement, we add it to our list of measurements
            if isinstance(operation, UnitaryOperation):
                op_start = int(operation.t0)
                # TODO: Duration should be some property of the operation
                op_end = op_start + operation.discrete_steps
                temp_coeffs = coeffs_dict[f"x{operation.channel}"]
                # TODO: Check the logic for more complicated circuits
                coeffs = np.append(
                    temp_coeffs[0:op_start], operation.data["pulse_info"][0]["samples"]
                )
                coeffs = np.append(coeffs, temp_coeffs[op_end : len(temp_coeffs)])
                coeffs_dict[f"x{operation.channel}"] = coeffs
            elif isinstance(operation, MeasurementOperation):
                # TODO: How do we store times in the database?
                measurements.append((operation.channel, int(operation.t0 * 10e8)))
        t0 = datetime.now()

        measurement_timestamps = list(map(lambda x: x[1], measurements))
        self.processor.set_coeffs(coeffs_dict)
        self.processor.set_tlist(tlist_dict)
        simulation_data = self.processor.run_state(
            init_state=initial_state, tlist=measurement_timestamps
        )
        # wait for program to finish and return acquisition
        repetitions = 1024  # TODO: move to some configuration

        # We need qubit id, acquisition index, and sampled data

        results = {}

        i_q_mapping = self.backend_config["simulator"]["i_q_mapping"]
        cov_matrix = self.backend_config["simulator"]["cov_matrix"]

        # We iterate over the simulation data states
        for (
            i_,
            (channel_, _),
        ) in enumerate(measurements):
            # To skip the initial measurement at time 0, which is necessary in qutip
            if channel_ is not None:
                sampling_state = (
                    project_on_qubit(simulation_data.states[i_]).ptrace(channel_).full()
                )
                # We take the diagonal from the state to sample
                sampling_probabilities = np.array(
                    [abs(sampling_state[0, 0]), abs(sampling_state[1, 1])]
                )
                # We normalize the sampling probabilities
                normalized_sampling_probabilities = sampling_probabilities / np.sum(
                    sampling_probabilities
                )
                # We sample the result from the given probabilities
                sample = np.random.choice(
                    [0, 1], size=repetitions, p=normalized_sampling_probabilities
                )
                # Translate the collapsed state vector to I and Q values
                i_q_values = np.array(
                    [
                        np.random.multivariate_normal(
                            i_q_mapping[channel_][s_], np.array(cov_matrix), 1
                        )[0]
                        for s_ in sample
                    ]
                )
                # Cast to complex values
                complex_i_q_values = np.array([complex(*x_) for x_ in i_q_values])
                results[channel_] = (
                    ["repetition", f"acq_index_{channel_}"],
                    np.array(complex_i_q_values).reshape(-1, 1),
                )
        # Now the dataset contains the discriminated values already, but we want the I Q values

        result_dataset = Dataset(results)
        print(f"{results=}")
        t1 = datetime.now()
        print(t1 - t0, "DURATION OF SIMULATION")
        return result_dataset

    def close(self):
        pass
