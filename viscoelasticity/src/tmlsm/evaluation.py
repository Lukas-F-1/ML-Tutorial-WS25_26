"""
evaluation.py

Modular evaluation utilities for viscoelastic models in this repository.

Design principles:
- Each function does ONE job.
- No hidden global state.
- Works with models defined in models.py (Simple RNN, Maxwell, Maxwell+NN, GSM).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from . import metrics as tm_metrics


# ============================================================================
# Data containers
# ============================================================================

@dataclass(frozen=True)
class Trajectory:
    """Container for a single loadcase trajectory."""
    eps: np.ndarray          # (T,)
    dts: np.ndarray          # (T,)
    sig: np.ndarray          # (T,)
    eps_dot: Optional[np.ndarray] = None  # (T,) if available


@dataclass(frozen=True)
class SimulationResult:
    """Model simulation output for one loadcase."""
    gamma: np.ndarray   # (T+1,) including initial gamma_0
    sig: np.ndarray     # (T,)


# ============================================================================
# Core simulation (crucial building block)
# ============================================================================

def simulate_model(model, eps, x2):
    eps_j = jnp.asarray(eps)
    x2_j  = jnp.asarray(x2)
    xs = jnp.stack([eps_j, x2_j], axis=1)  # (T,2)

    def scan_fn(state, x):
        return model.cell(state, x)

    init_state = jnp.array(0.0)
    _, ys = jax.lax.scan(scan_fn, init_state, xs)

    # For your models, ys is sigma history (T,) because scan returns (_, y)
    # But in our previous version we unpacked (gamma_new, sig). Let's do explicit:
    def scan_fn2(gamma, x):
        gamma_new, sig = model.cell(gamma, x)
        return gamma_new, (gamma_new, sig)

    _, (gamma_hist, sig_hist) = jax.lax.scan(scan_fn2, init_state, xs)
    gamma_full = jnp.concatenate([init_state[None], gamma_hist], axis=0)

    return gamma_full, sig_hist


def simulate_model_batch(model, eps_batch, x2_batch):
    def _one(eps, x2):
        return simulate_model(model, eps, x2)

    gamma_b, sig_b = jax.vmap(_one)(eps_batch, x2_batch)
    return np.array(gamma_b), np.array(sig_b)




# ============================================================================
# Error arrays + scalar metrics
# ============================================================================

def stress_error(sig_true: np.ndarray, sig_pred: np.ndarray) -> np.ndarray:
    """Pointwise error e(t) = sig_pred - sig_true."""
    return np.asarray(sig_pred) - np.asarray(sig_true)


def abs_stress_error(sig_true: np.ndarray, sig_pred: np.ndarray) -> np.ndarray:
    """Pointwise absolute error |e(t)|."""
    return np.abs(stress_error(sig_true, sig_pred))


def compute_metrics_per_case(sig_true_batch: np.ndarray, sig_pred_batch: np.ndarray) -> Dict[int, Dict[str, float]]:
    """
    Compute standard metrics per loadcase (index -> metrics dict).

    Shapes:
        sig_true_batch: (N, T)
        sig_pred_batch: (N, T)
    """
    out: Dict[int, Dict[str, float]] = {}
    for i in range(sig_true_batch.shape[0]):
        out[i] = tm_metrics.compute_all_metrics(sig_true_batch[i], sig_pred_batch[i])
    return out


def summarize_metrics(metrics_per_case: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """Average metrics over loadcases."""
    keys = list(next(iter(metrics_per_case.values())).keys())
    return {k: float(np.mean([m[k] for m in metrics_per_case.values()])) for k in keys}


# ============================================================================
# Hysteresis / dissipation proxies from sigma-epsilon
# ============================================================================

def hysteresis_area(eps: np.ndarray, sig: np.ndarray) -> float:
    """
    Approximate loop area ∮ sigma d eps.

    For a full closed cycle (harmonic steady state), this is dissipated energy per cycle.
    For non-closed paths, it is net work over the path.
    """
    eps = np.asarray(eps)
    sig = np.asarray(sig)
    # numerical line integral: ∫ sigma d eps
    return float(np.trapz(sig, eps))


def hysteresis_area_batch(eps_batch: np.ndarray, sig_batch: np.ndarray) -> np.ndarray:
    """Compute hysteresis area per loadcase."""
    return np.array([hysteresis_area(eps_batch[i], sig_batch[i]) for i in range(eps_batch.shape[0])])


# ============================================================================
# Energy & dissipation for specific model families
# ============================================================================

def maxwell_energy(eps: np.ndarray, gamma: np.ndarray, E_infty: float, E: float) -> np.ndarray:
    """
    e(eps,gamma) = 0.5 E_infty eps^2 + 0.5 E (eps - gamma)^2
    gamma should be aligned with eps. If eps has length T, use gamma[0:T].
    """
    eps = np.asarray(eps)
    gamma = np.asarray(gamma)
    g = gamma[: len(eps)]
    return 0.5 * E_infty * eps**2 + 0.5 * E * (eps - g) ** 2


def maxwell_gamma_dot(eps: np.ndarray, gamma: np.ndarray, E: float, eta: float) -> np.ndarray:
    """
    gamma_dot = (E/eta) * (eps - gamma)
    (continuous form; discrete evaluation at time steps)
    """
    eps = np.asarray(eps)
    g = np.asarray(gamma)[: len(eps)]
    return (E / eta) * (eps - g)


def maxwell_dissipation_density(eps: np.ndarray, gamma: np.ndarray, E: float, eta: float) -> np.ndarray:
    """
    D = (E^2/eta) * (eps - gamma)^2  >= 0
    """
    eps = np.asarray(eps)
    g = np.asarray(gamma)[: len(eps)]
    return (E**2 / eta) * (eps - g) ** 2


# ---------------- Maxwell + NN: extract f_theta ----------------------------

def maxwell_nn_f(model: Any, eps: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    Extract f_theta(eps,gamma) from MaxwellNNCell by re-running its MLP.
    This relies on the structure in your models.py: model.cell.layers + activations.

    Returns f evaluated at each time step n, aligned with eps_n and gamma_n.
    """
    cell = model.cell
    if not hasattr(cell, "layers") or not hasattr(cell, "activations"):
        raise ValueError("Model does not look like MaxwellNNModel (missing layers/activations).")

    eps_j = jnp.asarray(eps)
    gamma_j = jnp.asarray(gamma[: len(eps)])

    def f_one(e, g):
        x = jnp.array([e, g])
        for layer, act in zip(cell.layers, cell.activations):
            x = act(layer(x))
        return x[0]

    f_vals = jax.vmap(f_one)(eps_j, gamma_j)
    return np.array(f_vals)


def maxwell_nn_gamma_dot(model: Any, eps: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    gamma_dot = f_theta(eps,gamma) * (eps - gamma)
    """
    f = maxwell_nn_f(model, eps, gamma)
    g = np.asarray(gamma)[: len(eps)]
    return f * (np.asarray(eps) - g)


def maxwell_nn_dissipation_density(model: Any, eps: np.ndarray, gamma: np.ndarray, E: float) -> np.ndarray:
    """
    D = E * (eps - gamma) * gamma_dot = E * f * (eps - gamma)^2
    Requires E (non-equilibrium spring stiffness).
    """
    f = maxwell_nn_f(model, eps, gamma)
    g = np.asarray(gamma)[: len(eps)]
    return E * f * (np.asarray(eps) - g) ** 2


# ---------------- GSM: access learned energy + gradients --------------------

def gsm_energy(model: Any, eps: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    Evaluate learned energy e_theta(eps,gamma) using model.cell._energy.
    """
    cell = model.cell
    if not hasattr(cell, "_energy"):
        raise ValueError("Model does not look like GSMModel (missing _energy).")

    eps_j = jnp.asarray(eps)
    gamma_j = jnp.asarray(gamma[: len(eps)])

    e_vals = jax.vmap(cell._energy)(eps_j, gamma_j)
    return np.array(e_vals)


def gsm_de_dgamma(model: Any, eps: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Compute de/dgamma along a trajectory."""
    cell = model.cell
    if not hasattr(cell, "_energy"):
        raise ValueError("Model does not look like GSMModel (missing _energy).")

    eps_j = jnp.asarray(eps)
    gamma_j = jnp.asarray(gamma[: len(eps)])

    de_dg_fn = jax.grad(cell._energy, argnums=1)
    vals = jax.vmap(de_dg_fn)(eps_j, gamma_j)
    return np.array(vals)


def gsm_dissipation_density(model: Any, eps: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    D = g * (de/dgamma)^2  >= 0
    """
    cell = model.cell
    if not hasattr(cell, "g"):
        raise ValueError("Model does not look like GSMModel (missing g).")

    de_dg = gsm_de_dgamma(model, eps, gamma)
    return float(cell.g) * de_dg**2


# ============================================================================
# Plotting helpers (modular)
# ============================================================================

from typing import Dict, Optional, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt


def plot_multi_model_predictions(
    eps_batch: np.ndarray,
    sig_true_batch: np.ndarray,
    sig_pred_by_model: Dict[str, np.ndarray],
    omegas: Sequence[float],
    As: Sequence[float],
    title: str = "Model comparison",
    cases: Optional[Union[int, Sequence[int]]] = None,
    t: Optional[np.ndarray] = None,
) -> None:
    """
    Plot GT + multiple model predictions, but *one figure per loadcase*.

    Parameters
    ----------
    eps_batch : (N,T)
    sig_true_batch : (N,T)
    sig_pred_by_model : dict[name -> (N,T)]
        Each entry must have same shape as sig_true_batch.
    omegas, As : length N
        Metadata used in titles.
    title : str
        Base title prefix.
    cases : None | int | list[int]
        Which loadcases to plot.
        - None: plot all loadcases (one figure per case)
        - int: plot that case
        - list/tuple: plot those cases
    t : optional np.ndarray (T,)
        If None, uses linspace(0, 2π, T) (harmonic assumption).
        If you want true time, pass cumulative sum of dt externally.
    """
    eps_batch = np.asarray(eps_batch)
    sig_true_batch = np.asarray(sig_true_batch)

    n_cases, T = eps_batch.shape

    # Validate shapes
    if sig_true_batch.shape != (n_cases, T):
        raise ValueError(f"sig_true_batch must have shape {(n_cases, T)}, got {sig_true_batch.shape}")

    for name, sig_pred in sig_pred_by_model.items():
        sig_pred = np.asarray(sig_pred)
        if sig_pred.shape != (n_cases, T):
            raise ValueError(f"sig_pred_by_model['{name}'] must have shape {(n_cases, T)}, got {sig_pred.shape}")

    # Normalize cases argument
    if cases is None:
        case_list = list(range(n_cases))
    elif isinstance(cases, int):
        case_list = [cases]
    else:
        case_list = list(cases)

    # bounds check
    for c in case_list:
        if c < 0 or c >= n_cases:
            raise IndexError(f"case index {c} out of bounds for N={n_cases}")

    # Time axis
    if t is None:
        ts = np.linspace(0, 2 * np.pi, T)
    else:
        ts = np.asarray(t)
        if ts.shape != (T,):
            raise ValueError(f"t must have shape {(T,)}, got {ts.shape}")

    # Plot one figure per case
    for i in case_list:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{title} — case {i} (ω={omegas[i]}, A={As[i]})")

        # sigma vs time
        ax = axs[0]
        ax.plot(ts, sig_true_batch[i], "--", alpha=0.8, linewidth=2.0, label="GT")
        for name, sig_pred in sig_pred_by_model.items():
            ax.plot(ts, sig_pred[i], "-", linewidth=1.8, label=name)

        ax.set_xlabel("time t")
        ax.set_ylabel("stress σ")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

        # sigma vs eps (hysteresis)
        ax = axs[1]
        ax.plot(eps_batch[i], sig_true_batch[i], "--", alpha=0.8, linewidth=2.0, label="GT")
        for name, sig_pred in sig_pred_by_model.items():
            ax.plot(eps_batch[i], sig_pred[i], "-", linewidth=1.8, label=name)

        ax.set_xlabel("strain ε")
        ax.set_ylabel("stress σ")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

        fig.tight_layout()
        plt.show()



def plot_error_vs_time(
    sig_true_batch: np.ndarray,
    sig_pred_batch: np.ndarray,
    title: str = "Stress error vs time",
) -> None:
    n_cases, T = sig_true_batch.shape
    ts = np.linspace(0, 2 * np.pi, T)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(title)

    for i in range(n_cases):
        err = sig_pred_batch[i] - sig_true_batch[i]
        ax.plot(ts, err, label=f"case {i}")

    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("time t")
    ax.set_ylabel("σ_pred - σ_true")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_abs_error_vs_strain(
    eps_batch: np.ndarray,
    sig_true_batch: np.ndarray,
    sig_pred_batch: np.ndarray,
    title: str = "Absolute stress error vs strain",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(title)

    n_cases = eps_batch.shape[0]
    for i in range(n_cases):
        err = np.abs(sig_pred_batch[i] - sig_true_batch[i])
        ax.scatter(eps_batch[i], err, s=8, alpha=0.6, label=f"case {i}")

    ax.set_xlabel("strain ε")
    ax.set_ylabel("|σ_pred - σ_true|")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_abs_error_vs_strain_rate(
    eps_dot_batch: np.ndarray,
    sig_true_batch: np.ndarray,
    sig_pred_batch: np.ndarray,
    title: str = "Absolute stress error vs strain rate",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(title)

    n_cases = eps_dot_batch.shape[0]
    for i in range(n_cases):
        err = np.abs(sig_pred_batch[i] - sig_true_batch[i])
        ax.scatter(eps_dot_batch[i], err, s=8, alpha=0.6, label=f"case {i}")

    ax.set_xlabel("strain rate ε̇")
    ax.set_ylabel("|σ_pred - σ_true|")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_energy_and_dissipation(
    ts: np.ndarray,
    energy_by_model: Dict[str, np.ndarray],
    diss_by_model: Dict[str, np.ndarray],
    title: str = "Energy and dissipation",
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)

    ax = axs[0]
    for name, e in energy_by_model.items():
        ax.plot(ts, e, label=name)
    ax.set_xlabel("time t")
    ax.set_ylabel("energy e")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axs[1]
    for name, d in diss_by_model.items():
        ax.plot(ts, d, label=name)
    ax.set_xlabel("time t")
    ax.set_ylabel("dissipation density D")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    fig.tight_layout()
    plt.show()
