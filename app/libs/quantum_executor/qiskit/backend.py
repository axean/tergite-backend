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

import datetime
import numpy as np

import jax

from qiskit.providers.models import PulseBackendConfiguration, GateConfig, PulseDefaults
from qiskit_dynamics import DynamicsBackend, Solver
from qiskit.transpiler import Target, InstructionProperties
from qiskit.providers import QubitProperties
from qiskit.circuit import Delay, Reset, Parameter
from qiskit.circuit.library import RZGate, SXGate, XGate

# configure jax to use 64 bit mode
jax.config.update("jax_enable_x64", True)
# configure jax to use 64 bit mode
jax.config.update("jax_platform_name", "cpu")


class FakeOpenPulse1Q(DynamicsBackend):
    r"""Backend for pulse simulations on a single transmon qubit.

    Args:
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
        f: float = 4.7e9,
        alpha: float = -0.17e9,
        t1: float = 71e-6,
        t2: float = 69e-6,
        r: float = 1e9,
        dt: float = 1e-9,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        dim: int = 4,
        noise: bool = True,
        **options,
    ):
        backend_name = "fake_openpulse_1q"
        backend_version = "1.0.0"

        a = np.diag(np.sqrt(np.arange(1, dim)), 1)  # annihilation operator
        adag = np.diag(np.sqrt(np.arange(1, dim)), -1)  # creation operator
        N = np.diag(np.arange(dim))  # number operator
        ident = np.eye(dim, dtype=complex)

        # Create static Hamiltonian
        static_ham = 2 * np.pi * f * N + np.pi * alpha * N * (N - ident)

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

        configuration = PulseBackendConfiguration(
            backend_name=backend_name,
            backend_version=backend_version,
            n_qubits=1,
            basis_gates=["x"],
            gates=GateConfig(
                name="x",
                parameters=[],
                qasm_def="gate x q { U(pi, 0, pi) q; }",
                coupling_map=[[0]],
            ),
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=True,
            memory=True,
            max_shots=4000,
            coupling_map=[],
            meas_map=[[0]],
            n_uchannels=0,
            u_channel_lo=[],
            meas_levels=[1, 2],
            qubit_lo_range=[[f / 1e9 - 0.1, f / 1e9 + 0.1]],  # in GHz
            meas_lo_range=[],
            dt=dt / 1e-9,  # in nanoseconds
            dtm=dt / 1e-9,  # in nanoseconds
            rep_times=[],
            meas_kernels=["boxcar"],
            discriminators=["max_1Q_fidelity"],
            description="A single transmon Hamiltonian with 4 levels",
            hamiltonian={
                "h_str": [
                    "2*np.pi*f*N0",
                    "np.pi*alpha*N0*N0",
                    "-np.pi*alpha*N0",
                    "X0||D0",
                ],
                "vars": {"f": f, "alpha": alpha},
                "qub": {"0": dim},
                "osc": {},
                "description": "A single transmon Hamiltonian with 4 levels",
            },
        )

        # Not sure if this information is needed or used.
        defaults = PulseDefaults(
            qubit_freq_est=[f / 1e9],
            meas_freq_est=[0],
            buffer=0,
            pulse_library=[],
            cmd_def=[],
            meas_kernel=None,
            discriminator=None,
        )

        target = Target(
            num_qubits=1,
            qubit_properties=[QubitProperties(frequency=f, t1=t1, t2=t2)],
            dt=dt,
            granularity=1,
        )

        self.solver = Solver(
            static_hamiltonian=static_ham,
            hamiltonian_operators=[drive_op],
            rotating_frame=static_ham,
            hamiltonian_channels=["d0"],
            channel_carrier_freqs={"d0": f},
            static_dissipators=static_dissipators,
            dt=dt,
            array_library="numpy",
        )

        solver_options = {
            "method": "jax_odeint",
            "atol": atol,
            "rtol": rtol,
            "hmax": dt,
        }

        super().__init__(
            solver=self.solver,
            target=target,
            subsystem_dims=[dim],
            solver_options=solver_options,
            configuration=configuration,
            defaults=defaults,
            **options,
        )

    @property
    def target(self):
        """Contains information for circuit transpilation."""
        return self._target

    def to_db(self):
        num_qubits = self.configuration().num_qubits

        qubit_properties = []
        resonator_properties = []

        for i in range(num_qubits):
            qubit_properties.append(
                {
                    "id": i,
                    "frequency": self.qubit_properties(i).frequency,
                    "pi_pulse_amplitude": 0.014248841224281961,
                    "pi_pulse_duration": 56e-9,
                    "pulse_type": "Gaussian",
                    "pulse_sigma": 7e-9,
                    "t1_decoherence": self.qubit_properties(i).t1,
                    "t2_decoherence": self.qubit_properties(i).t2,
                }
            )
            resonator_properties.append(
                {
                    "id": i,
                    "acq_delay": 0,
                    "acq_integration_time": 0,
                    "frequency": 0,
                    "pulse_amplitude": 0,
                    "pulse_delay": 0,
                    "pulse_duration": 0,
                    "pulse_type": "Square",
                }
            )

        backend_db_schema = {
            "name": self.configuration().backend_name,
            "characterized": True,
            "open_pulse": self.configuration().open_pulse,
            "timelog": {
                "CREATED": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            },
            "version": self.configuration().backend_version,
            "meas_map": self.configuration().meas_map,
            "coupling_map": self.configuration().coupling_map,
            "description": self.configuration().description,
            "simulator": self.configuration().simulator,
            "num_qubits": self.configuration().num_qubits,
            "num_couplers": 0,
            "num_resonators": 1,
            "online_date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "dt": self.configuration().dt,
            "dtm": self.configuration().dtm,
            "qubit_ids": [f"q{i_}" for i_ in range(num_qubits)],
            "device_properties": {
                "qubit": qubit_properties,
                "readout_resonator": resonator_properties,
            },
            "discriminators": "TODO",
            "gates": "TODO",
        }
        return backend_db_schema
