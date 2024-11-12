# This code is part of Tergite
#
# (C) Chalmers Next Labs (2024)
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np

# FIXME: Does this require some copyright stuff?


def delta_t_function(t, args):
    """
    Vectorized envelope function δ(t) as a function of time.

    Returns:
    - delta_t: the value of δ(t) (same shape as t)
    """
    t_w = args["t_w"]
    t_rf = args["t_rf"]
    t_p = args["t_p"]
    delta_0 = args["delta_0"]

    condlist = [
        t <= t_w,
        (t > t_w) & (t <= t_w + t_rf / 2),
        (t > t_w + t_rf / 2) & (t < t_w + t_rf / 2 + t_p),
        (t >= t_w + t_rf / 2 + t_p) & (t < t_w + t_rf + t_p),
        t >= t_w + t_rf + t_p,
    ]
    choicelist = [
        0,
        delta_0 / 2 * (1 - np.cos(2 * np.pi * (t - t_w) / t_rf)),
        delta_0,
        delta_0 / 2 * (1 - np.cos(2 * np.pi * (t - t_w - t_p) / t_rf)),
        0,
    ]
    return np.select(condlist, choicelist)


def flux(t, args):
    """
    Coupled flux Φ(t) as a function of time, dependent on delta_t_function.

    Returns:
    - Phi(t): flux value (same shape as t)
    """
    theta = args["Theta"]
    omega_phi = args["omega_Phi"]
    phi = args["phi"]

    # Get the envelope delta_t from the delta_t_function
    delta_t = delta_t_function(t, args)

    # Compute the flux Φ(t) using delta_t
    return theta + delta_t * np.cos(omega_phi * t + phi)


def omega_c(t, args):
    """
    Coupler frequency as a function of time, dependent on Phi_t which itself depends on delta_t_function.

    Parameters:
    - t: time (scalar or array)
    - args: dict

    Returns:
    - omega_c(t): the coupler frequency (same shape as t)
    """
    omega_c0 = args["omega_c0"]

    # Compute Phi(t) using Phi_t which itself depends on delta_t_function
    phi_t_vals = flux(t, args)

    # Return omega_c using the computed Phi(t)
    return omega_c0 * np.sqrt(np.abs(np.cos(np.pi * phi_t_vals)))
