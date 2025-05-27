# This code is part of Tergite
#
# (C) Pontus Vikstål, Stefan Hill (2024)
# (C) Martin Ahindura (2025)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import abc
from typing import Any, Dict, List, Optional

from qiskit.providers.models import PulseBackendConfiguration, PulseDefaults
from qiskit.transpiler import Target
from qiskit_dynamics import DynamicsBackend, Solver

from app.libs.device_parameters import BackendConfig


class QiskitPulseBackend(DynamicsBackend):
    """Backend for running simulators on using QiskitDynamics"""

    def __init__(
        self,
        backend_config: BackendConfig,
        alpha: float = -0.17e9,
        r: float = 1e9,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        dim: int = 4,
        noise: bool = True,
        **options
    ):
        self.backend_config = backend_config
        self.backend_name = backend_config.general_config.name

        kwargs = self.__get_dynamic_backend_kwargs(
            alpha=alpha,
            r=r,
            atol=atol,
            rtol=rtol,
            dim=dim,
            noise=noise,
            backend_config=backend_config,
            **options
        )

        super().__init__(**kwargs, **options)

    @classmethod
    def __get_dynamic_backend_kwargs(cls, **kwargs):
        """Generates the kwargs for initializing the parent DynamicBackend class

        Args:
            kwargs: the initial key-word arguments fed in during initialization

        Returns:
            a dictionary of options to pass to the dynamic backend on initialization
        """
        options = dict(
            solver=cls.generate_solver(**kwargs),
            target=cls.generate_target(**kwargs),
            solver_options=cls.generate_solver_options(**kwargs),
            configuration=cls.generate_configuration(**kwargs),
            defaults=cls.generate_pulse_defaults(**kwargs),
            subsystem_dims=cls.generate_subsystem_dims(**kwargs),
        )

        # clean out any None values
        return {k: v for k, v in options.items() if v is not None}

    @classmethod
    @abc.abstractmethod
    def generate_solver(cls, **kwargs) -> Solver:
        """Generates the solver to pass to the backend when initializing"""
        pass

    @classmethod
    @abc.abstractmethod
    def generate_configuration(cls, **kwargs) -> Optional[PulseBackendConfiguration]:
        """Generates the PulseBackendConfiguration to pass to the backend when initializing"""
        pass

    @classmethod
    @abc.abstractmethod
    def generate_target(cls, **kwargs) -> Optional[Target]:
        """Generates the Target to pass to the backend when initializing"""
        pass

    @classmethod
    @abc.abstractmethod
    def generate_solver_options(cls, **kwargs) -> Dict[str, Any]:
        """Generates the solver options to pass to the backend when initializing"""
        pass

    @classmethod
    @abc.abstractmethod
    def generate_pulse_defaults(cls, **kwargs) -> Optional[PulseDefaults]:
        """Generates the PulseDefault's to pass to the backend when initializing"""
        pass

    @classmethod
    @abc.abstractmethod
    def generate_subsystem_dims(cls, **kwargs) -> Optional[List[int]]:
        """Generates the subsystem_dims to pass to the backend when initializing"""
        pass

    @abc.abstractmethod
    def train_discriminator(self, shots: int = 1024, **kwargs):
        """
        Generates |0> and |1> states, trains a linear discriminator
        Args:
            shots: number of shots for generating i q data
            kwargs: extra key-word args

        Returns:
            Discriminator object as json in the format to store it in the database
        """
        pass
