# This code is part of Tergite
#
# (C) Martin Ahindura (2024)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Base class for all simulators

We expect to have multiple simulators, each using different techniques.
They, however, should all satisfy the BaseSimulator class
"""
from abc import abstractmethod
from typing import Any

from quantify_scheduler import CompiledSchedule, Schedule
from xarray import Dataset


class BaseSimulator:
    """The base class for all simulators"""

    @abstractmethod
    def run(
        self: object,
        schedule: CompiledSchedule,
        /,
        *,
        output: str = "voltage_single_shot",
    ) -> Dataset:
        """Runs a given compiled schedule on the simulator.

        Args:
            schedule: The QuantifyCore schedule to run.
            output (str, optional): Mode of output, either 'voltage' which returns complex voltages,
                or 'raw' which returns measurement probabilites. Defaults to 'voltage_single_shot'.
        """
        raise NotImplementedError("run() method not implemented for this simulator")

    @abstractmethod
    def compile(self: object, schedule: Schedule, /) -> CompiledSchedule:
        """Compiles a schedule which has been translated by quantify to be run on the simulator.

        Args:
            schedule: The schedule to compile.
        """
        raise NotImplementedError("compile() method not implemented for this simulator")

    @abstractmethod
    def generate_backend_config(self: object, /) -> dict[str, Any]:
        """
        Returns the device backend config for the backend class in tergite.qiskit client
        """
        raise NotImplementedError(
            "generate_backend_config() method not implemented for this simulator"
        )
