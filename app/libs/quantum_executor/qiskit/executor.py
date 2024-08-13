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

import jax
import numpy as np
import xarray
from qiskit.qobj import PulseQobj
from qiskit_dynamics import Solver, DynamicsBackend

import app.libs.storage_file as storagefile
from app.libs.quantum_executor.base.executor import QuantumExecutor
from app.libs.quantum_executor.base.experiment import BaseExperiment
from app.libs.quantum_executor.qiskit.experiment import QiskitDynamicsExperiment
from app.libs.quantum_executor.utils.channel import Channel
from app.libs.quantum_executor.utils.instruction import Instruction

# configure jax to use 64 bit mode
jax.config.update("jax_enable_x64", True)
# configure jax to use 64 bit mode
jax.config.update("jax_platform_name", "cpu")


class QiskitDynamicsExecutor(QuantumExecutor):
    def __init__(self, config_file):
        super().__init__()

        # -> Have in backend already
        dt = 1e-9

        # -> Number of energy levels (static)
        dim = 3

        # -> First qubit
        f0 = 4.8
        anharm0 = -0.17

        f1 = 7.8
        anharm1 = -0.12

        f2 = 4.225
        anharm2 = -0.18

        # coupling -> all qubits are coupled in a linear chain
        g = 0.07

        a = np.diag(np.sqrt(np.arange(1, dim)), 1)
        adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
        N = np.diag(np.arange(dim))

        ident = np.eye(dim, dtype=complex)
        full_ident = np.eye(dim**3, dtype=complex)

        # -> dynamic for each qubit
        N0 = np.kron(ident, np.kron(ident, N))
        N1 = np.kron(ident, np.kron(N, ident))
        N2 = np.kron(N, np.kron(ident, ident))

        a0 = np.kron(ident, np.kron(ident, a))
        a1 = np.kron(ident, np.kron(a, ident))
        a2 = np.kron(a, np.kron(ident, ident))

        a0dag = np.kron(ident, np.kron(ident, adag))
        a1dag = np.kron(ident, np.kron(adag, ident))
        a2dag = np.kron(adag, np.kron(ident, ident))

        static_ham0 = 2 * np.pi * f0 * N0 + np.pi * anharm0 * N0 * (N0 - full_ident)
        static_ham1 = 2 * np.pi * f1 * N1 + np.pi * anharm1 * N1 * (N1 - full_ident)
        static_ham2 = 2 * np.pi * f2 * N2 + np.pi * anharm2 * N2 * (N2 - full_ident)

        # -> Convert frequency into radial frequency
        coupling_ham = (
            2
            * np.pi
            * g
            * (((a0dag + a0) @ (a1dag + a1)) + ((a2dag + a2) @ (a1dag + a1)))
        )

        static_ham = static_ham0 + static_ham1 + static_ham2
        static_ham_full = static_ham + coupling_ham

        # First qubit to second qubit plus the coupler
        drive_op0 = a0 + a0dag
        drive_op1 = a1 + a1dag
        ctrl_op = N1

        # Channels
        self.solver = Solver(
            static_hamiltonian=static_ham_full,
            hamiltonian_operators=[drive_op0, drive_op1, ctrl_op],
            rotating_frame=static_ham,
            hamiltonian_channels=["d0", "d1", "u0"],
            channel_carrier_freqs={"d0": f0, "d1": f2, "u0": 0},
            dt=dt,
            array_library="jax",
        )

        solver_options = {
            "method": "jax_odeint",
            "atol": 1e-6,
            "rtol": 1e-8,
            "hmax": dt,
        }

        self.backend = DynamicsBackend(
            solver=self.solver,
            subsystem_dims=[dim, dim, dim],  # for computing measurement data
            solver_options=solver_options,  # to be used every time run is called
            meas_level=1,
            meas_return="single",
        )

    def run(self, experiment: BaseExperiment, /) -> xarray.Dataset:
        job = self.backend.run(experiment.schedule)
        result = job.result()
        return result.data()["memory"]

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

    def close(self):
        pass
