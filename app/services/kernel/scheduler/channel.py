# This code is part of Tergite
#
# (C) Axel Andersson (2022)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Refactored by Martin Ahindura (2024)

from dataclasses import dataclass

from ..utils.general import find


@dataclass(eq=True, frozen=False)
class Channel:
    clock: str
    frequency: float = 0.0
    phase: float = 0.0
    acquisitions: int = 0

    def __hash__(self: "Channel") -> int:
        return hash(self.clock)


def default_frequency(clock: str, hw_config: dict) -> float:
    """Returns the default LO frequency of a clock sequencer's mixer.
    Return value is in units Hz. Depends on the hardware config key structure.
    """

    # find all the complex output dictionaries
    for path in find(hw_config, "portclock_configs"):
        output = hw_config
        for k in path[:-1]:
            output = output[k]

        # test if clock is in portclocks configuration
        arr = [pc for pc in output["portclock_configs"] if clock in pc.values()]
        if len(arr) > 0:
            # if found clock, return lo freq of that mixer
            if "lo_freq" in output.keys():
                return output["lo_freq"]

    return 0.0
