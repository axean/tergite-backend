# This code is part of Tergite
#
# (C) Pontus Vikstål, Stefan Hill (2024)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np

import jax

from qiskit.providers.models import PulseBackendConfiguration, GateConfig, PulseDefaults
from qiskit.quantum_info import Statevector
from qiskit_dynamics import DynamicsBackend, Solver
from qiskit.transpiler import Target, InstructionProperties
from qiskit.providers import QubitProperties
from qiskit.circuit import Delay, Reset, Parameter
from qiskit.circuit.library import XGate, SXGate, RZGate
from qiskit.pulse import Acquire, AcquireChannel, MemorySlot, Schedule
import datetime

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from app.libs.properties import BackendConfig

# configure jax to use 64 bit mode
jax.config.update("jax_enable_x64", True)
# configure jax to use 64 bit mode
jax.config.update("jax_platform_name", "cpu")


class FakeOpenPulse1Q(DynamicsBackend):
    r"""Backend for pulse simulations on a single transmon qubit.

    Args:
        backend_config: the configuration of this backend
        f (float): The qubit frequency in Hz. Default is 4.7e9.
        alpha (float): The qubit anharmonicity in Hz. Default is -0.17e9.
        t1 (float): The qubit T1 relaxation time in seconds. Default is 71e-8.
        t2 (float): The qubit T2 dephasing time in seconds. Default is 69e-8.
        r (float): The rabi rate in Hz. Default is 1e9.
        dt (float): The timestep for the pulse simulation in seconds. Default is 1e-9.
        atol (float): The absolute tolerance for the numerical solver. Default is 1e-6.
        rtol (float): The relative tolerance for the numerical solver. Default is 1e-6.
        dim (int): The dimension of the Hilbert space for the system. Default is 4.
        noise (bool): Flag to enable or disable noise in the simulation. Default is True.
        **options: Additional keyword arguments passed to the DynamicsBackend.

    """

    def __init__(
        self,
        backend_config: BackendConfig,
        alpha: float = -0.17e9,
        r: float = 1e9,
        # atol: float = 1e-6,
        # rtol: float = 1e-6,
        dim: int = 4,
        noise: bool = True,
        **options,
    ):
        backend_name = backend_config.general_config.name
        backend_version = backend_config.general_config.version
        dt = backend_config.general_config.dt

        self.backend_config = backend_config
        self.backend_name = backend_config.general_config.name
        # This is for a single qubit backend
        first_qubit_conf = backend_config.simulator_config.qubit[0]
        freq = first_qubit_conf["frequency"]
        t1 = first_qubit_conf["t1_decoherence"]
        t2 = first_qubit_conf["t2_decoherence"]

        a = np.diag(np.sqrt(np.arange(1, dim)), 1)  # annihilation operator
        adag = np.diag(np.sqrt(np.arange(1, dim)), -1)  # creation operator
        N = np.diag(np.arange(dim))  # number operator
        ident = np.eye(dim, dtype=complex)

        # Create static Hamiltonian
        static_ham = 2 * np.pi * freq * N + np.pi * alpha * N * (N - ident)

        # Create drive operator
        drive_op = 2 * np.pi * r * (a + adag)

        # Setup static dissipators
        if noise:
            static_dissipators = []
            t2_eff = 1.0 / (1.0 / t2 - 1.0 / 2.0 / t1)
            static_dissipators.append(1 / np.sqrt(t1) * a)
            static_dissipators.append(1 / np.sqrt(2 * t2_eff) * 2 * N)
        else:
            static_dissipators = None

        gates_configs = [
            GateConfig(name=k, **v) for k, v in backend_config.gates.items()
        ]

        configuration = PulseBackendConfiguration(
            backend_name=backend_name,
            backend_version=backend_version,
            n_qubits=backend_config.general_config.num_qubits,
            basis_gates=list(backend_config.gates.keys()),
            gates=gates_configs[0],
            local=True,
            simulator=backend_config.general_config.simulator,
            conditional=False,
            open_pulse=backend_config.general_config.open_pulse,
            memory=True,
            max_shots=4000,
            coupling_map=backend_config.device_config.coupling_map,
            meas_map=backend_config.device_config.meas_map,
            n_uchannels=0,
            u_channel_lo=[],
            meas_levels=[1, 2],
            qubit_lo_range=[[freq / 1e9 - 0.1, freq / 1e9 + 0.1]],  # in GHz
            meas_lo_range=[],
            dt=dt / 1e-9,  # in nanoseconds
            dtm=dt / 1e-9,  # in nanoseconds
            rep_times=[],
            meas_kernels=["boxcar"],
            discriminators=["max_1Q_fidelity"],
            description=backend_config.general_config.description,
            hamiltonian={
                "h_str": [
                    "2*np.pi*f*N0",
                    "np.pi*alpha*N0*N0",
                    "-np.pi*alpha*N0",
                    "X0||D0",
                ],
                "vars": {"f": freq, "alpha": alpha},
                "qub": {"0": dim},
                "osc": {},
                "description": "A single transmon Hamiltonian with 4 levels",
            },
        )

        # Not sure if this information is needed or used.
        defaults = PulseDefaults(
            qubit_freq_est=[freq / 1e9],
            meas_freq_est=[0],
            buffer=0,
            pulse_library=[],
            cmd_def=[],
            meas_kernel=None,
            discriminator=None,
        )

        target = Target(
            num_qubits=1,
            qubit_properties=[QubitProperties(frequency=freq, t1=t1, t2=t2)],
            dt=dt,
            granularity=1,
        )
        for instruction in (Reset(), RZGate(Parameter("angle")), SXGate(), XGate()):
            target.add_instruction(
                instruction,
                properties={(0,): InstructionProperties(duration=0)},
            )
        target.add_instruction(Delay(Parameter("duration")))

        solver = Solver(
            static_hamiltonian=static_ham,
            hamiltonian_operators=[drive_op],
            rotating_frame=static_ham,
            hamiltonian_channels=["d0"],
            channel_carrier_freqs={"d0": freq},
            static_dissipators=static_dissipators,
            dt=dt,
            array_library="numpy",
        )

        # FIXME: I wonder why this is not used.
        # solver_options = {
        #     "method": "jax_odeint",
        #     "atol": atol,
        #     "rtol": rtol,
        #     "hmax": dt,
        # }

        super().__init__(
            solver=solver,
            target=target,
            subsystem_dims=[dim],
            solver_options={},
            configuration=configuration,
            defaults=defaults,
            **options,
        )

    @property
    def target(self):
        """Contains information for circuit transpilation."""
        return self._target

    def train_discriminator(self, shots: int = 1024):
        """
        Generates |0> and |1> states, trains a linear discriminator
        Args:
            shots: number of shots for generating i q data

        Returns:
            Discriminator object as json in the format to store it in the database

        """
        # Generate the iq values
        schedule = Schedule((0, Acquire(1, AcquireChannel(0), MemorySlot(0))))

        job_0 = self.run([schedule], shots=shots)
        i_q_values_0 = job_0.result().data()["memory"].reshape(shots, 2)
        job_1 = self.run(
            [schedule], shots=shots, initial_state=Statevector([0, 1, 0, 0])
        )
        i_q_values_1 = job_1.result().data()["memory"].reshape(shots, 2)

        # Train scikit learn discriminator
        combined_i_q_values = np.vstack((i_q_values_0, i_q_values_1))
        labels = np.append(np.zeros(shots), np.ones(shots))

        lda_model = LinearDiscriminantAnalysis()
        lda_model.fit(combined_i_q_values, labels)

        # Bring it to the right format
        return {
            "discriminators": {
                "lda": {
                    qubit_id: {
                        "intercept": float(lda_model.intercept_),
                        "coef_0": float(lda_model.coef_[0][0]),
                        "coef_1": float(lda_model.coef_[0][1]),
                    } for qubit_id in self.backend_config.device_config.qubit_ids
                }
            }
        }
