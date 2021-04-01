from itertools import tee

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.fft import rfft, rfftfreq
from scipy.cluster import hierarchy
from scipy.signal import argrelmax, stft
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f_oneway, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.tsa.stattools import acf
from scipy.stats import kurtosis, skew
from scripts.utils import *


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_largest_local_max(signal1D: np.ndarray, order: int = 1):
    """Return the largest local max and the associated index in a tuple.

    This function uses `order` points on each side to use for the comparison.
    """
    all_local_max_indexes = argrelmax(signal1D, order=order)[0]
    all_local_max = np.take(signal1D, all_local_max_indexes)
    largest_local_max_index = all_local_max_indexes[
        all_local_max.argsort()[-1]
    ]

    return signal1D[largest_local_max_index], largest_local_max_index


def fig_ax(figsize=(15, 5)):
    return plt.subplots(figsize=figsize)


def compute_auto_features_1D(signal_1D, n_lags=400, n_bins=100):
    """
    Compute features from a signal, as done in TP2.
    """
    signal = scale(signal_1D)

    FREQUENCY = 100  # sampling frequency

    res_dict = dict()
    # distribution features
    res_dict["mean"] = signal.mean()
    res_dict["std"] = signal.std()
    res_dict["min"] = signal.min()
    res_dict["max"] = signal.max()
    res_dict["kurtosis"] = kurtosis(signal)
    res_dict["skew"] = skew(signal)
    res_dict["25p"] = np.percentile(signal, 25)
    res_dict["50p"] = np.percentile(signal, 50)
    res_dict["75p"] = np.percentile(signal, 75)

    # auto-correlation feautres
    auto_corr = acf(signal, nlags=n_lags, fft=True)
    res_dict = dict()
    for (lag, auto_corr_value) in enumerate(auto_corr):
        res_dict[f"autocorrelation_{lag}_lag"] = auto_corr_value

    local_max, local_argmax = get_largest_local_max(auto_corr, order=20)  # 20
    res_dict["lag_max_autocorrelation_Hz"] = FREQUENCY / local_argmax
    res_dict["max_autocorrelation"] = local_max

    # spectral features
    n_bins = 100
    n_samples = signal.shape[0]
    fourier = abs(rfft(signal))
    freqs = rfftfreq(n=n_samples, d=1.0 / FREQUENCY)

    freq_bins = np.linspace(0, FREQUENCY / 2, n_bins + 1)
    for (f_min, f_max) in pairwise(freq_bins):
        keep = (f_min <= freqs) & (freqs < f_max)
        res_dict[f"fourier_{f_min:.1f}-{f_max:.1f}_Hz"] = np.log(np.sum(fourier[keep] ** 2))

    return res_dict


def compute_auto_features(signal_MD, n_lags=400, n_bins=100):
    """
    :param signal_MD: array, multidimensional signal, with shape (n_dim, n_time_steps)
    """
    n_dim, n_time_steps = signal_MD.shape
    assert n_dim < n_time_steps, "Too many dimensions"
    dicts = []
    for dim in range(n_dim):
        dicts.append(compute_auto_features_1D(signal_MD[dim], n_lags, n_bins))
    arrays = [np.array(list(d.values())) for d in dicts]
    return np.hstack(arrays)