# This code is part of Tergite
#
# (C) Copyright Simon Genne, Arvid Holmqvist, Bashar Oumari, Jakob Ristner,
#               Björn Rosengren, and Jakob Wik 2022 (BSc project)
# (C) Copyright Fabian Forslund, Nicklas Botö 2022
# (C) Copyright Abdullah-Al Amin 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from datetime import datetime, timezone

from archive.backend_properties_updater.Coupler import Coupler
from archive.backend_properties_updater.Gate import Gate
from archive.backend_properties_updater.Qubit import Qubit
from archive.backend_properties_updater.Resonator import Resonator


class config_updater:
    def __init__(
        self,
        qubit_configs,
        resonator_configs,
        gates,
        coupling_map,
    ):
        self.qubits = [
            Qubit(qubit_config, coupling_map) for qubit_config in qubit_configs
        ]
        self.resonators = [Resonator(resonator) for resonator in resonator_configs]
        self.couplers = [
            Coupler(coupler_config, coupling_map) for coupler_config in coupling_map
        ]
        self.gates = [Gate(gate) for gate in gates]

        self.last_update_date = datetime.now(timezone.utc)

    def config_update(self) -> None:
        self.last_update_date = datetime.now(timezone.utc)
        for qubit in self.qubits:
            qubit.update()
        for resonator in self.resonators:
            resonator.update()
        for coupler in self.couplers:
            coupler.update()
        for gate in self.gates:
            gate.update()

    def to_dict(self) -> dict:

        return {
            "last_update_date": self.last_update_date.isoformat(),
            "qubits": [qubit.to_dict() for qubit in self.qubits],
            "resonators": [resonator.to_dict() for resonator in self.resonators],
            "gates": [gate.to_dict() for gate in self.gates],
            "couplers": [coupler.to_dict() for coupler in self.couplers],
        }
