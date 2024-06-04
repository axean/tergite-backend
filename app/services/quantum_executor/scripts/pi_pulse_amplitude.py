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

import numpy as np


def calibrate_pi_pulse_amplitude():
    """
    This is to analytically calibrate the pi pulse amplitude.

    Returns
    -------

    """

    sigma = 7
    T = 56
    mu = T / 2

    def f_(t_):
        # Exp[-(t - \[Mu]) ^ 2 / (2 \[Sigma] ^ 2)];
        return np.exp(-(t_ - mu) ** 2 / (2 * sigma ** 2))

    amplitude = np.pi / (2 * np.trapz([f_(x_) for x_ in range(T)]))
    return amplitude


if __name__ == '__main__':
    pi_pulse_amplitude = calibrate_pi_pulse_amplitude()
    print(pi_pulse_amplitude)
