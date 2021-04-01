"""
Many functions in this code are inspired from the TP1 of the class "Machine Learning for Time-series analysis"
Master MVA
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from dtw import dtw
from IPython.display import Audio, display
from loadmydata.load_uea_ucr import load_uea_ucr_data
from matplotlib.colors import rgb2hex
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sporco import plot, util
from sporco.admm import cbpdn
from sporco.dictlrn import cbpdndl
from scripts.utils import *


def plot_CDL(signal, Z, D, figsize=(15, 10)):
    """Plot the learned dictionary `D` and the associated sparse codes `Z`.

    `signal` is an univariate signal of shape (n_samples,) or (n_samples, 1).
    """
    (atom_length, n_atoms) = np.shape(D)
    plt.figure(figsize=figsize)
    plt.subplot(n_atoms + 1, 3, (2, 3))
    plt.plot(signal)
    for i in range(n_atoms):
        plt.subplot(n_atoms + 1, 3, 3 * i + 4)
        plt.plot(D[:, i])
        plt.subplot(n_atoms + 1, 3, (3 * i + 5, 3 * i + 6))
        plt.plot(Z[:, i])
        plt.ylim((np.min(Z), np.max(Z)))


# In this cell, we set parameters and options that should probably remained unchanged
PENALTY = 3


# options for the dictionary learning and sparse coding procedures
def get_opt_dl(penalty=PENALTY):
    """Return the option class for the dictionary learning"""
    return cbpdndl.ConvBPDNDictLearn.Options(
        {
            "Verbose": False,
            "MaxMainIter": 50,
            "CBPDN": {"rho": 50.0 * penalty + 0.5, "NonNegCoef": True},
            "CCMOD": {"rho": 10.0},
        },
        dmethod="cns",
    )


def get_opt_sc():
    """Return the option class for the sparse coding"""
    return cbpdn.ConvBPDN.Options(
        {
            "Verbose": False,
            "MaxMainIter": 50,
            "RelStopTol": 5e-3,
            "AuxVarObj": False,
            "NonNegCoef": True,  # only positive sparse codes
        }
    )


def display_distance_matrix_as_table(
        distance_matrix, labels=None, figsize=(8, 2)
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("tight")
    ax.axis("off")
    norm = mpl.colors.Normalize()
    cell_colours_hex = np.empty(shape=distance_matrix.shape, dtype=object)
    cell_colours_rgba = plt.get_cmap("magma")(norm(distance_matrix))

    for i in range(distance_matrix.shape[0]):
        for j in range(i + 1, distance_matrix.shape[0]):
            cell_colours_hex[i, j] = rgb2hex(
                cell_colours_rgba[i, j], keep_alpha=True
            )
            cell_colours_hex[j, i] = cell_colours_hex[i, j]

    if labels is not None:
        _ = ax.table(
            cellText=distance_matrix,
            colLabels=labels,
            rowLabels=labels,
            loc="center",
            cellColours=cell_colours_hex,
        )
    else:
        _ = ax.table(
            cellText=distance_matrix,
            loc="center",
            cellColours=cell_colours_hex,
        )


def get_n_largest(
        arr: np.ndarray, n_largest: int = 3
) -> (np.ndarray, np.ndarray):
    """Return the n largest values and associated indexes of an array.

    (In decreasing order of value.)
    """
    indexes = np.argsort(arr)[-n_largest:][::-1]
    if n_largest == 1:
        indexes = np.array(indexes)
    values = np.take(arr, indexes)
    return values, indexes


def fig_ax(figsize=(15, 5)):
    return plt.subplots(figsize=figsize)


def learn_dict(signal, atom_length, n_atoms, rng, penalty=PENALTY):
    """
    Return the learned dictionaty

    :param signal: 2d_array with shape (n_time_steps, n_dims)
    :param atom_length: int, length of an atom
    :param n_atoms: int, number of atoms
    :param rng: random number generator
    :param penalty: float, sparsity penalty (Convolutional dictionary learning)
    """
    opt_dl = get_opt_dl()
    if signal.ndim == 1:
        signal = atleast_2d(signal)
    dimK = signal.shape[1]
    return cbpdndl.ConvBPDNDictLearn(
        D0=rng.randn(atom_length, dimK, n_atoms),  # random init; set to 2 here for multidimensionality
        S=signal,  # signal at hand
        dimK=dimK,  # set to 2 here for multidimensionality
        lmbda=penalty,  # sparsity penalty
        opt=opt_dl,  # options for the optimizations
        xmethod="admm",  # optimization method (sparse coding)
        dmethod="cns",  # optimization method (dict learnin)
    ).solve()


def learn_codes(signal, atom_dictionary, penalty=PENALTY):
    """Return the sparse codes"""
    opt_sc = get_opt_sc()
    return (
        cbpdn.ConvBPDN(
            D=atom_dictionary,  # learned dictionary
            S=atleast_2d(signal),  # signal at hand
            lmbda=penalty,  # sparsity penalty
            opt=opt_sc,  # options for the optimizations
        ).solve().squeeze())


def compute_error(signal, atom_dictionary, codes):
    """Return the MSE for the given dictionary and codes"""
    atom_length = atom_dictionary.shape[0]
    reconstruction = np.sum(
        [
            np.convolve(code, atom, mode="valid")
            for (code, atom) in zip(codes.T, atom_dictionary.squeeze().T)
        ],
        axis=0,
    )
    return np.mean((signal[atom_length - 1:].flatten() - reconstruction) ** 2)


def get_training_signal(sample, label, n_samples=2):
    """
    Return a sample for training.

    :param sample: dictionnary containing all the information about the recording of a night of a participant
    See output of DataLoader for more information.
    :param label: int, label number
    :param n_samples: int, number of samples of 30 sec to use to learn dictionary
    """
    indexes = annotated_sample(label, sample['labels'], n_samples=n_samples)
    signal = sample['data'][indexes].T
    signal = np.vstack([signal[:, :, k] for k in range(signal.shape[-1])])
    signal = scale(signal.T).T
    return atleast_2d(signal)
