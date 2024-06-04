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

from typing import List, Union

from quantify_scheduler import Operation


class SimulationSchedule:

    def __init__(self,
                 name: str = ""):
        self._name: str = name
        self._operations: List[BaseOperation] = []

    def add(self,
            operation: 'BaseOperation'):
        self._operations.append(operation)

    @property
    def operations(self) -> List['BaseOperation']:
        return self._operations

    @operations.setter
    def operations(self, value: List['BaseOperation']):
        self._operations = value

    @property
    def discrete_steps(self):
        # TODO: this function has to be properly tested
        t0_max = 0
        for operation in self.operations:
            if operation.t0 >= t0_max and isinstance(operation, UnitaryOperation):
                t0_max += operation.discrete_steps
        return t0_max


class BaseOperation(Operation):

    def __init__(self,
                 channel: Union[str, int],
                 t0: Union[int, float] = 0):
        super().__init__("")  # Takes name as input
        self.channel = channel
        self.t0 = t0


class UnitaryOperation(BaseOperation):

    def __init__(self,
                 *args,
                 frequency=0.0,
                 phase=0.0,
                 amp=0.0,
                 sigma=1,
                 discrete_steps=0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.frequency = frequency
        self.phase = phase
        self.amp = amp
        self.sigma = sigma
        self.discrete_steps = discrete_steps


class MeasurementOperation(BaseOperation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
