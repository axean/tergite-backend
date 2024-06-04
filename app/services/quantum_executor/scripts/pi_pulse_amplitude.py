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
