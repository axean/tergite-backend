# This code is part of Tergite
#
# (C) Copyright Abdullah Al Amin 2022
# (C) Copyright David Wahlstedt 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import Any, List, Dict

import settings
from pathlib import Path

import numpy as np
import scipy.signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from resonator_tools import circuit

import Labber

import qtanalysis.core as qc
import qtanalysis.utilities as qu

POSTPROC_PLOTTING = settings.POSTPROC_PLOTTING

def fit_resonator(logfile):
    labber_logfile = Labber.LogFile(logfile)

    # Labber api function, y value is not used scince it is not well defined, for
    # multiple y traces, y represents only the last one, not usable in this case
    # only x value is used for sweeped frequency array
    x, y = labber_logfile.getTraceXY()

    # Extracting data using qtl-analysis repo, xdict contains third dimension of
    # data, for example sweeped power values, ydict contains values of the traces
    xdict, ydict = qu.file_handling.LabberParsing(logfile)
    resonators_res_freq = extract_resonance_freqs(x, y, xdict, ydict)
    print("\n")
    print(f"Number or resonators: {len(resonators_res_freq[0])}")
    print(f"Number of traces: {len(xdict['x0']['values'])}")
    print(f"Corresponding power: {xdict['x0']['values']}")
    print(f"Resonance frequencies: {resonators_res_freq}")
    print("\n")
    return resonators_res_freq

# Processing results for resonators resonace frequencies
# algorithm for approximate peak detection is not robast
# may fail in some measurement/traces, improvement required.
# parameter y and xdict is not used, passed for possible
# future uses.
def extract_resonance_freqs(x, y, xdict, ydict):
    resonance_freqs = []
    for indx, traces in enumerate(ydict["y0"]["values"]):
        trace = np.absolute(traces)
        # Detrain the trace using scipy.signal
        trace = scipy.signal.detrend(
            trace, axis=-1, type="linear", bp=0, overwrite_data=False
        )
        # Filter noise to smooth the trace, second parameter in the
        # savgol_filter needs to be odd, higher value of it results in
        # more smothing, but at the cost of reducing resosnance dip
        trace = scipy.signal.savgol_filter(trace, 11, 2)
        # Used for defining peak detection criterion
        mean = np.mean(trace)
        std = np.std(trace)
        # FIXME: Quick hard coded solution, improvment required
        min_height = -3 * std - mean
        # Detecting the possible and approximate pick or dips
        peaks, _ = scipy.signal.find_peaks(-trace, height=-min_height,
                                           distance=100)
        # Extracting the frequiencies of the peaks
        peak_freqs = x[peaks]
        # Each list in resonance_freq list represents corresponding
        # resonanace frequiencies of the current trace
        resonance_freqs.append(peak_freqs.tolist())

    return resonance_freqs


def fit_resonator_idx(logfile: Path, idxs) -> List[float]:
    labber_logfile = Labber.LogFile(logfile)

    x, y = labber_logfile.getTraceXY()
    xdict, ydict = qu.file_handling.LabberParsing(logfile)

    results = []

    for p_idx in idxs:
        # indices refer to measurements at corresponding powers
        trace = ydict["y0"]["values"][p_idx]
        port1 = circuit.notch_port(x, trace)
        port1.autofit()
        results += [port1.fitresults]
        if POSTPROC_PLOTTING:
            port1.plotall()

    return results


def gaussian_fit_idx(logfile: Path, idxs) -> List[float]:

    xdict, ydict = qu.file_handling.LabberParsing(logfile)

    results = []

    def gauss(x, H, A, x0, sigma):
        return H + A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    def gauss_fit(x, y):
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
        popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
        return popt

    # idxs points to the trace number for multiple recordings in the same log
    for p_idx in idxs:
        freq_array = xdict["x0"]["values"]
        trace = np.absolute(ydict["y0"]["values"][p_idx])

        H, A, x0, sigma = gauss_fit(freq_array, trace)
        FWHM = 2.35482 * sigma

        print("The offset of the gaussian baseline is", H)
        print("The center of the gaussian fit is", x0)
        print("The sigma of the gaussian fit is", sigma)
        print("The maximum intensity of the gaussian fit is", H + A)
        print("The Amplitude of the gaussian fit is", A)
        print("The FWHM of the gaussian fit is", FWHM)

        fitted_data = gauss(freq_array, *gauss_fit(freq_array, trace))

        results += [x0]

        if POSTPROC_PLOTTING:
            # plotting of the fitting, may be useful later
            plt.plot(freq_array, trace, "--b", label="orginal data")
            plt.plot(freq_array, fitted_data, "-r", label="fit")
            plt.legend()
            plt.title("Gaussian fit,  $f(x) = A e^{(-(x-x_0)^2/(2sigma^2))}$")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude (V)")
            plt.show()

    return results

def fit_oscillation_idx(logfile: Path, idxs) -> List[Dict[str, float]]:

    xdict, ydict = qu.file_handling.LabberParsing(logfile)

    results = []

    def fit_sin(x, y):
        ff = np.fft.fftfreq(len(x), (x[1] - x[0]))  # assume uniform spacing
        ffy = abs(np.fft.fft(y))
        guess_freq = abs(
            ff[np.argmax(ffy[1:]) + 1]
        )  # excluding the zero frequency "peak", which is related to offset
        guess_amp = np.std(y) * 2.0**0.5
        guess_offset = np.mean(y)
        guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])

        def sinfunc(t, A, w, p, c):
            return A * np.sin(w * t + p) + c

        popt, pcov = scipy.optimize.curve_fit(sinfunc, x, y, p0=guess)
        A, w, p, c = popt
        f = w / (2.0 * np.pi)
        fitfunc = lambda t: A * np.sin(w * t + p) + c
        return {
            "amp": A,
            "omega": w,
            "phase": p,
            "offset": c,
            "freq": f,
            "period": 1.0 / f,
            "fitfunc": fitfunc,
            "maxcov": np.max(pcov),
            "rawres": (guess, popt, pcov),
        }

    # idxs points to the trace number for multiple recordings in the same log
    for p_idx in idxs:
        freq_array = xdict["x0"]["values"]
        trace = np.absolute(ydict["y0"]["values"][p_idx])
        res = fit_sin(freq_array, trace)
        # In case of Rabi, the period is the inverted the frequency
        results += [res]
        print(
            "Amplitude=%(amp)s, freq=%(freq)s, period.=%(period)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s"
            % res
        )

        if POSTPROC_PLOTTING:
            plt.plot(freq_array, trace, "-k", label="y", linewidth=2)
            plt.plot(
                freq_array,
                res["fitfunc"](freq_array),
                "r-",
                label="y fit curve",
                linewidth=2,
            )
            plt.show()

    return results
