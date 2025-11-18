# visualization.py

import numpy as np
import matplotlib.pyplot as plt

def plot_F_diagonals(F, time=None, title="Diagonal entries of F over time"):
    """
    Plot F11, F22 and F33 as functions of 'time'.

    Parameters
    ----------
    F : ndarray, shape (N, 3, 3)
        Sequence of deformation gradients. Each F[i] is a 3x3 matrix.
    time : array-like, optional
        1D array of length N giving the time (or load step index) for each F.
        If None, the function uses np.arange(N).
    title : str, optional
        Title of the plot.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """
    F = np.asarray(F)
    if F.ndim != 3 or F.shape[1:] != (3, 3):
        raise ValueError("F must have shape (N, 3, 3)")

    n_steps = F.shape[0]

    if time is None:
        time = np.arange(n_steps)
    else:
        time = np.asarray(time)
        if time.shape[0] != n_steps:
            raise ValueError("time must have length equal to F.shape[0]")

    # Extract diagonal components
    F11 = F[:, 0, 0]
    F22 = F[:, 1, 1]
    F33 = F[:, 2, 2]

    fig, ax = plt.subplots()
    ax.plot(time, F11, label="F11")
    ax.plot(time, F22, label="F22")
    ax.plot(time, F33, label="F33")

    ax.set_xlabel("Step / time")
    ax.set_ylabel("Diagonal entries of F")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    return fig, ax
