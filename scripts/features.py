"""
This file contains methods to compute features from Polysomnography
"""

import mne
import os
import matplotlib.pyplot as plt
import nolds
import numpy as np
import pandas as pd


# FRACTAL DIMENSION
def rolling_ar(x, window, steps):
    """
    Generate a view of the input array, for a window size and a stride (i.e. steps) given.
    Inspired from https://medium.com/analytics-vidhya/a-thorough-understanding-of-numpy-strides-and-its-application-in-data-processing-e40eab1c82fe
    :param x: array
    :param window: int, width of the sliding window
    :param steps: int, stride of sliding window
    """
    total_len_x = x.shape[-1]
    new_shape_row = (total_len_x - window) // steps + 1
    new_shape_col = window
    new_shape = (new_shape_row, new_shape_col)
    n_bytes = x.strides[-1]

    stride_steps_row = n_bytes * steps
    stride_steps_col = n_bytes
    stride_steps = (stride_steps_row, stride_steps_col)
    y = np.lib.stride_tricks.as_strided(x, new_shape, stride_steps)
    return y


def fractal_dimension(ts, n_max=20, plot=False):
    """
    Compute fractal dimension.
    :param ts: array, time-series
    :param n_max: int, max length of the trajectories
    :param plot: boolean, if True, plot sequence of (ln(L),(S(L)/L))
    """
    N = len(ts)
    x_ = np.log(np.arange(1, n_max + 1))
    y_ = np.zeros(n_max)
    for n in range(1, n_max + 1):
        rolled_ar = rolling_ar(ts, window=n + 1, steps=n)
        S = np.sum(np.max(rolled_ar, axis=-1) - np.min(rolled_ar, axis=-1))
        y_[n - 1] = np.log(S / n)
    # plot for checking
    if plot:
        plt.title("Sequence (ln(L),(S(L)/L))")
        plt.scatter(x_, y_, marker='+')
        plt.xlabel('ln(L)')
        plt.ylabel('ln(S(L)/L)')
    # linear regression
    x_ = np.stack([x_, np.ones(n_max)]).T
    (p, r1, r2, s) = np.linalg.lstsq(x_, y_, rcond=None)
    return -p[0]


def higuchi_fractal_dimension(a, k_max=20):
    """
    https://stackoverflow.com/questions/47259866/fractal-dimension-algorithms-gives-results-of-2-for-time-series
    """
    L = []
    x = []
    N = len(a)

    for k in range(1, k_max):
        Lk = 0
        for m in range(0, k):
            # we pregenerate all idxs
            idxs = np.arange(1, int(np.floor((N - m) / k)), dtype=np.int32)
            Lmk = np.sum(np.abs(a[m + idxs * k] - a[m + k * (idxs - 1)]))
            Lmk = (Lmk * (N - 1) / (((N - m) / k) * k)) / k
            Lk += Lmk

        L.append(np.log(Lk / (m + 1)))
        x.append([np.log(1.0 / k), 1])

    (p, r1, r2, s) = np.linalg.lstsq(x, L, rcond=None)
    return p[0]


def katz_fractal_dimension(data):
    """
    https://stackoverflow.com/questions/47259866/fractal-dimension-algorithms-gives-results-of-2-for-time-series
    """
    n = len(data) - 1
    L = np.hypot(np.diff(data), 1).sum()  # Sum of distances
    d = np.hypot(data - data[0], np.arange(len(data))).max()  # furthest distance from first point
    return np.log10(n) / (np.log10(d / L) + np.log10(n))


# DETRENDED FLUCTUATION ANALYSIS
def dfa_sasha(ts, debug_plot=False, n_scales=15):
    """
    Perform detrended fluctuation analysis.

    :param ts: array, inputted time-series
    :param debug_plot: boolean, if True plot regression
    :param n_scales: int, number of points (log(L),log(f(L)))

    :return : float, scaling exponent DFA-alpha
    """
    nvals = None # np.logspace(3, np.log10(len(ts) // 100), n_scales).astype(int) # to modify (ts are epochs, so much smaller)
    return nolds.dfa(ts, nvals=nvals, order=1, overlap=False, debug_plot=debug_plot)


# SHANNON ENTROPY
def shannon_entropy(ts,n_boxes=100):
    """
    Compute shannon entropy: entropy of amplitudes, as defined in the article "Discrimination of sleep stages".
    :param ts: array, inputted time-series
    :param n_boxes: int, number of equidistant boxes covering the interval between max and min amplitude of the EEG segment

    :return float, shannon entropy
    """
    hist, bin_edges = np.histogram(ts,bins=n_boxes)
    hist = hist/hist.sum()
    hist = hist[hist>0]
    return -np.sum(hist*np.log2(hist))/np.log2(n_boxes)


# APPROXIMATE ENTROPY
def approximate_entropy(ts, m, r=None):
    """
    Compute approximate entropy.
    :param ts: array, inputted time-series
    :param m: int, pattern length
    :param r: float, error threshold

    :return float, sample entropy
    """
    x = ts
    N = len(x)
    B = 0.0
    A = 0.0

    if r is None:
        r = 0.1 * np.std(x)

    # Split time series and save all templates of length m
    xmi = np.array([x[i: i + m] for i in range(N - m)])
    xmj = np.array([x[i: i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi]) / (N - m + 1)

    # Similar for computing A
    m += 1
    xm = np.array([x[i: i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm]) / (N - m)

    # Return ApEn
    return np.log(B) / (N - m + 1) - np.log(A) / (N - m)


def ApEn(U, m=2, r=None):
    """
    Compute approximate entropy of the signal U.

    :param U: array, time-series
    :param m: int, length of the patterns
    :param r: error threshold

    :return float, approximate entropy
    """
    U = np.array(U)
    N = U.shape[0]

    if r is None:
        r = np.std(U) * 0.2

    def _phi(m):
        z = N - m + 1.0
        x = np.array([U[i:i + m] for i in range(int(z))])
        X = x[:, np.newaxis]
        C = np.sum(np.absolute(x - X).max(axis=2) <= r, axis=0) / z
        return np.log(C).sum() / z

    return abs(_phi(m + 1) - _phi(m))


# SAMPLE ENTROPY
def sample_entropy(ts, m, r=None):
    """
    Compute sample entropy.
    :param ts: array, inputted time-series
    :param m: int, pattern length
    :param r: float, error threshold

    :return float, sample entropy

    Reference:
    Code adapted https://en.wikipedia.org/wiki/Sample_entropy
    """
    x = ts
    N = len(x)

    if r is None:
        r = 0.1 * np.std(x)

    # Split time series and save all templates of length m
    xmi = np.array([x[i: i + m] for i in range(N - m)])
    xmj = np.array([x[i: i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([x[i: i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return Sample Entropy
    return -np.log(A / B)


# MULTISCALE ENTROPY
def multiscale_entropy(ts, m, tau, r=None):
    """
    Compute multiscale entropy.
    :param ts: array, inputted time-series
    :param m: int, pattern length
    :param tau: scale scale factor
    :param r: float, error threshold

    :return array, multi-scale entropy (sample entropy at different scales)
    """
    x = ts
    if r is None:
        r = 0.1 * np.std(x)

    n = x.shape[0]

    k = int(np.floor(n / tau))
    y = x[:tau * k].reshape((k, tau))
    y = y.mean(axis=1)
    me = sample_entropy(y, m, r)

    return me


# DETRENDED FLUCTUATION
def dfa(ts, min_L, max_L):
    """
    Compute detrended fluctuation analysis alpha coefficient
    :param ts: time series
    :param min_L: minimum time scale
    :param max_L: maximum time scale

    :return float, dfa-alpha
    """

    x = ts
    n = x.shape[0]
    rmse = np.zeros(int(max_L - min_L + 1))
    L_s = np.arange(min_L, max_L + 1)

    for i in range(len(L_s)):
        L = L_s[i]
        k = int(np.floor(n / L))

        # Integrated time series
        y = np.cumsum(x - np.mean(x))
        y = y[:int(k * L)].reshape((k, L))

        # Linear fitting
        x_scale = np.arange(L)
        coef = np.polyfit(x_scale, y.T, 1)
        trend = (coef[0][:, None] * x_scale + coef[1][:, None])

        y = y.reshape(-1)
        trend = trend.reshape(-1)

        # RMSE F(L)
        rmse[i] = np.sqrt(np.mean((y - trend) ** 2))

    # Linear fitting (log(L), log(F(L)))
    coef = np.polyfit(np.log10(L_s), np.log10(rmse), 1)
    alpha = coef[0]

    return alpha