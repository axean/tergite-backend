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

from .base import QuantumExecutor
from .executors.quantify import QuantifyExecutor
from .executors.qutip import QuTipExecutor
from .executors.scqt import SCQTQuantifyExecutor


class QuantumExecutorFactory:
    """
    Factory pattern to load the executor implementations as specified in the environmental configuration
    """

    def __init__(self):
        # This is the map of configuration names for the executors to class implementations
        self._executors = {
            'hardware': QuantifyExecutor,
            'scqt': SCQTQuantifyExecutor,
            'qutip': QuTipExecutor
        }

    def get_executor(self, executor_name: str) -> 'QuantumExecutor':
        """
        Parameters
        ----------
        executor_name: str
            Executor name as in the configuration file

        Returns
        -------
        QuantumExecutor
            An implementation of a QuantumExecutor class
        """
        if executor_name not in self._executors.keys():
            raise KeyError(f"Executor with name: '{executor_name}' not implemented."
                           f"Please check the value of your EXECUTOR_TYPE variable in the environment."
                           f"EXECUTOR_TYPE can be: {self._executors.keys()}")
        return self._executors[str.lower(executor_name)]


