from __future__ import annotations

import jax
import jax.numpy as jnp
from . import data_t2 as td2
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from . import eval_workflows as ewf
import re
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory
import numpy as np
import matplotlib.pyplot as plt


def evaluate_growth_condition(
    models_or_runs,
    *,
    model_type: str,
    dataset_1: dict | None = None,
    G_cub=None,
    n: int = 80,
    det_min: float = 1e-6,
    det_max: float = 1.0,
    include_identity: bool = True,
    path: str = "uniaxial_compression",  # "uniaxial_compression" or "isotropic"
    reduce: str = "mean",
    return_per_init: bool = True,
    det_max_large: float | None = None,  # e.g. 10.0, 100.0; if None no upper extension
    n_large: int | None = None,          # optional separate resolution for upper side

):
    """
    Evaluate the growth condition by probing a deformation family with det(F) -> 0^+.

    Generates log-spaced determinants in (det_min, det_max] and constructs F(t) such that det(F)=t.
    Optionally includes F=I explicitly.

    Returns dict compatible with updated plotting.
    """
    # ---- normalize models ----
    if isinstance(models_or_runs, (ewf.Run,)):
        models = [models_or_runs.model]
    elif isinstance(models_or_runs, (list, tuple)) and len(models_or_runs) > 0 and isinstance(models_or_runs[0], ewf.Run):
        models = [r.model for r in models_or_runs]
    elif callable(models_or_runs):
        models = [models_or_runs]
    elif isinstance(models_or_runs, (list, tuple)) and len(models_or_runs) > 0 and callable(models_or_runs[0]):
        models = list(models_or_runs)
    else:
        raise TypeError("models_or_runs must be a Run, list[Run], model callable, or list of model callables.")

    K = len(models)

    mt = model_type.strip().upper()
    if mt == "WI":
        if dataset_1 is not None and "G_ti" in dataset_1:
            G_ti = dataset_1["G_ti"]
        else:
            G_ti = jnp.array([[4.0, 0.0, 0.0],
                              [0.0, 0.5, 0.0],
                              [0.0, 0.0, 0.5]])
    elif mt == "WI_CUBIC":
        if G_cub is None:
            G_cub = td2.G_cub()
    elif mt == "WF":
        pass
    else:
        raise ValueError("model_type must be one of {'WI','WI_Cubic','WF'}.")

    # ---- determinants: log-spaced down to det_min ----
    # We include det_max in the sequence; det_max=1 corresponds to F=I for these paths.
    # ---- determinants: lower side (det_max -> det_min) log-spaced ----
    dets_low = jnp.geomspace(det_max, det_min, num=n)

    # ---- optional upper side (det_max -> det_max_large) log-spaced ----
    if det_max_large is not None:
        if det_max_large <= det_max:
            raise ValueError(f"det_max_large must be > det_max (got {det_max_large} <= {det_max}).")
        n_hi = int(n_large) if n_large is not None else int(max(10, n // 2))
        dets_high = jnp.geomspace(det_max, det_max_large, num=n_hi)
        # avoid duplicating det_max
        dets = jnp.concatenate([dets_low, dets_high[1:]], axis=0)
    else:
        dets = dets_low

    def _F_from_det(d):
        if path == "uniaxial_compression":
            # det(F)=d with F=diag(d,1,1)
            return jnp.array([[d, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]])
        elif path == "isotropic":
            # F = c I, det = c^3 -> c = d^(1/3)
            c = d ** (1.0 / 3.0)
            return jnp.array([[c, 0.0, 0.0],
                              [0.0, c, 0.0],
                              [0.0, 0.0, c]])
        else:
            raise ValueError("path must be 'uniaxial_compression' or 'isotropic'.")

    F_all = jnp.stack([_F_from_det(d) for d in dets], axis=0)  # (n,3,3)

    # Optionally ensure identity is explicitly present
    # For det_max=1 it already is; but this forces inclusion even if det_max != 1.
    identity_idx = None
    if include_identity:
        F_I = jnp.eye(3)
        # append F=I if not already essentially included
        if not (float(det_max) == 1.0):
            F_all = jnp.concatenate([F_all, F_I[None, :, :]], axis=0)
            dets = jnp.concatenate([dets, jnp.array([1.0])], axis=0)
            identity_idx = int(F_all.shape[0] - 1)
        else:
            # det_max==1 means first element is F=diag(1,1,1) for both paths
            identity_idx = 0

    # ---- evaluate W per init ----
    N = int(F_all.shape[0])
    W_per_init = np.zeros((K, N), dtype=float)

    for k, model in enumerate(models):
        vals = []
        for i in range(N):
            F = F_all[i]

            if mt == "WI":
                I = td2.compute_all_invariants(F=F, G_ti=G_ti)
                out = model((F, I))
            elif mt == "WI_CUBIC":
                I = td2.compute_all_invariants_cubic(F=F, G_cub=G_cub)
                out = model((F, I))
            else:  # WF
                out = model(F)

            W_pred = out[0] if (isinstance(out, tuple) and len(out) == 2) else out
            vals.append(float(jnp.squeeze(W_pred)))

        W_per_init[k, :] = np.array(vals, dtype=float)

    # ---- reduce across inits ----
    if reduce == "mean":
        W_red = W_per_init.mean(axis=0)
    elif reduce == "median":
        W_red = np.median(W_per_init, axis=0)
    else:
        raise ValueError("reduce must be 'mean' or 'median'.")

    W_std = W_per_init.std(axis=0)

    # ---- also compute ||F||_F for plotting convenience ----
    F_np = np.array(F_all)
    F_norm = np.linalg.norm(F_np.reshape(N, -1), axis=1)

    result = {
        "F_all": F_np,
        "detF": np.array(dets, dtype=float),
        "F_norm": np.array(F_norm, dtype=float),
        "W_mean": np.array(W_red, dtype=float),
        "W_std": np.array(W_std, dtype=float),
        "n_inits": K,
        "identity_idx": identity_idx,
        "path": path,
    }
    if return_per_init:
        result["W_per_init"] = W_per_init

    return result


def evaluate_normalization_condition(model, model_type: str):
    """
    Evaluate P(I) for a given trained model (normalization condition).
    
    Parameters
    ----------
    model : trained model object
    model_type : str
        "WI", "WI_Cubic", "WF", or "FFNN"
    
    Returns
    -------
    P_pred : jnp.ndarray
        Predicted first Piola-Kirchhoff stress tensor at the identity.
        Shape (3,3)
    """
    
    # Identity deformation gradient
    F = jnp.eye(3)

    # -------------------------------------------------------
    # Prepare input depending on model type
    # -------------------------------------------------------
    if model_type == "WI":
        # Compute TI invariants (I1, J, -J, I4, I5)
        I = td2.compute_all_invariants(F)  # shape (5,)
        model_input = I

        # Model outputs (W, P)
        W_pred, P_pred = model(model_input)

    elif model_type == "WI_Cubic":
        # Compute cubic invariants (I1, I2, J, -J, I7, I11)
        I = td2.compute_all_invariants_cubic_single(F)  # shape (6,)
        model_input = I

        W_pred, P_pred = model(model_input)

    elif model_type == "WF":
        # WF model takes F directly; computes cofF, detF internally
        W_pred, P_pred = model(F)

    elif model_type == "FFNN":
        # FFNN stress model takes C or similar directly.
        # Find your exact preprocessing; assuming C = F^T F:
        C = F.T @ F
        C_vec = C.flatten()  # depends on your FFNN input structure
        P_pred = model(C_vec)  # FFNN outputs stress directly

    else:
        raise ValueError(f"Unknown model_type '{model_type}'.")

    # -------------------------------------------------------
    # Ensure P is 3x3 matrix (FFNN may output vector)
    # -------------------------------------------------------
    if P_pred.ndim == 1 and P_pred.shape[0] == 9:
        P_pred = P_pred.reshape(3,3)

    return P_pred

def evaluate_multiple_observers(
    model,
    F_test,
    num_observers=10,
    key=jax.random.PRNGKey(0),
    mode="WF",
    G=None,   # only needed for WI models
):
    """
    Vectorized objectivity evaluation for WF and WI models.

    Parameters
    ----------
    model : callable
        WF model: model(F)              -> (W, P)
        WI model: model(F, I)           -> (W, P)

    F_test : (N, 3, 3)
        Deformation gradients.

    num_observers : int
        Number of random Q ∈ SO(3) rotations.

    mode : str
        "WF" or "WI".

    key : PRNGKey
        Random seed.

    G : (3,3) or (3,3,3,3)
        Structural/anisotropy tensor for invariant computation.
        Required in WI mode.

    Returns
    -------
    dict with mean/max errors for W and P.
    """

    assert mode in ("WF", "WI"), "mode must be 'WF' or 'WI'"

    # -----------------------------------------
    # 1) Generate rotation matrices
    # -----------------------------------------
    keys = jax.random.split(key, num_observers)
    Q = jax.vmap(td2.random_rotation)(keys)   # (obs, 3, 3)

    # -----------------------------------------
    # 2) Evaluate original model predictions
    # -----------------------------------------
    if mode == "WF":
        W0, P0 = jax.vmap(model)(F_test)  # (N,), (N,3,3)

    elif mode == "WI":
        assert G is not None, "G tensor required for WI invariants"
        I_test = td2.compute_all_invariants_cubic(F_test, G)
        W0, P0 = jax.vmap(lambda F, I: model((F, I)))(F_test, I_test)


    # -----------------------------------------
    # 3) Rotate F → QF  (obs, N, 3,3)
    # -----------------------------------------
    F_rot = jax.vmap(lambda q: q @ F_test)(Q)

    # -----------------------------------------
    # 4) Evaluate model on rotated F
    # -----------------------------------------
    if mode == "WF":
        # (obs, N), (obs, N,3,3)
        W_rot, P_rot = jax.vmap(lambda F_batch: jax.vmap(model)(F_batch))(F_rot)

    elif mode == "WI":
        # Compute invariants for all rotated F
        # Input: (obs, N, 3,3)
        compute_I_rot = jax.vmap(
            lambda Fbatch: td2.compute_all_invariants_cubic(Fbatch, G)
        )
        I_rot = compute_I_rot(F_rot)   # (obs, N, dimI)

        # Apply model to each observer batch
        # 1) apply model sample-wise
        apply_model = jax.vmap(lambda F, I: model((F, I)))

        # 2) apply over observers
        W_rot, P_rot = jax.vmap(apply_model)(F_rot, I_rot)



    # -----------------------------------------
    # 5) Expected rotated stresses Q P0
    # -----------------------------------------
    P_expected = jax.vmap(lambda q: q @ P0)(Q)  # (obs, N, 3,3)

    # -----------------------------------------
    # 6) Objectivity errors
    # -----------------------------------------
    W_err = jnp.abs(W_rot - W0)                   # (obs, N)
    P_err = jnp.linalg.norm(P_rot - P_expected, axis=(2,3))

    # -----------------------------------------
    # 7) Return summary statistics
    # -----------------------------------------
    return {
        "W_error_mean": float(jnp.mean(W_err)),
        "W_error_max":  float(jnp.max(W_err)),
        "P_error_mean": float(jnp.mean(P_err)),
        "P_error_max":  float(jnp.max(P_err)),
    }

def evaluate_objectivity(
    model,
    F_test,
    model_type: str,
    num_observers: int = 10,
    key=jax.random.PRNGKey(0),
    G=None,
):
    """
    Unified objectivity evaluation wrapper.

    Parameters
    ----------
    model : callable
        - WF model      ("WF"):        model(F)        -> (W, P)
        - WI cubic      ("WI_cubic"):  model((F, I))   -> (W, P)
        - FFNN with F   ("FFNN_F"):    model(F)        -> P
          (naive stress model, no energy)

    F_test : (N, 3, 3)
        Deformation gradients to test.

    model_type : str
        One of:
            "WF"        : deformation-based PANN W(F), P(F)
            "WI_cubic"  : cubic invariant-based PANN W(F), P(F)
            "FFNN_F"    : naive FFNN taking F as input, outputting P only

    num_observers : int
        Number of random rotations Q ∈ SO(3) (observers).

    key : PRNGKey
        Random seed for rotation sampling.

    G : tensor
        Structural / anisotropy tensor, required for "WI_cubic"
        (passed through to evaluate_multiple_observers).

    Returns
    -------
    dict
        {
          "W_error_mean": float or None,
          "W_error_max":  float or None,
          "P_error_mean": float,
          "P_error_max":  float,
        }
    """
    # ----------------------------------------------------------
    # Delegate to existing function for WF and WI_cubic
    # ----------------------------------------------------------
    if model_type == "WF":
        # W(F), P(F) with F as input
        return evaluate_multiple_observers(
            model=model,
            F_test=F_test,
            num_observers=num_observers,
            key=key,
            mode="WF",
            G=None,
        )

    if model_type == "WI_cubic":
        # invariant-based cubic model, W(F), P(F)
        assert G is not None, "G tensor must be provided for WI_cubic"
        return evaluate_multiple_observers(
            model=model,
            F_test=F_test,
            num_observers=num_observers,
            key=key,
            mode="WI",
            G=G,
        )

    # ----------------------------------------------------------
    # FFNN_F: naive FFNN taking F directly as input, outputting P
    # ----------------------------------------------------------
    if model_type == "FFNN_F":
        # 1) Sample rotations
        keys = jax.random.split(key, num_observers)
        Q = jax.vmap(td2.random_rotation)(keys)        # (obs, 3, 3)

        # 2) Baseline stresses P(F)
        P0_raw = jax.vmap(model)(F_test)               # (N,9) or (N,3,3)

        # Ensure P0 has shape (N,3,3)
        if P0_raw.ndim == 2 and P0_raw.shape[1] == 9:
            P0 = P0_raw.reshape(-1, 3, 3)
        elif P0_raw.ndim == 3 and P0_raw.shape[1:] == (3, 3):
            P0 = P0_raw
        else:
            raise ValueError(
                f"Unexpected FFNN_F output shape {P0_raw.shape}; "
                "expected (N,9) or (N,3,3)."
            )

        # 3) Rotate F → QF for each observer
        #    F_rot: (obs, N, 3,3)
        F_rot = jax.vmap(lambda q: q @ F_test)(Q)

        # 4) Model predictions for rotated F
        #    P_rot_raw: (obs, N, 9) or (obs, N, 3,3)
        P_rot_raw = jax.vmap(lambda F_batch: jax.vmap(model)(F_batch))(F_rot)

        # Ensure P_rot has shape (obs, N, 3,3)
        if P_rot_raw.ndim == 3 and P_rot_raw.shape[-1] == 9:
            P_rot = P_rot_raw.reshape(num_observers, -1, 3, 3)
        elif P_rot_raw.ndim == 4 and P_rot_raw.shape[-2:] == (3, 3):
            P_rot = P_rot_raw
        else:
            raise ValueError(
                f"Unexpected FFNN_F rotated output shape {P_rot_raw.shape}; "
                "expected (obs,N,9) or (obs,N,3,3)."
            )

        # 5) Expected rotated stresses Q P(F)
        #    P_expected: (obs, N, 3,3)
        P_expected = jax.vmap(lambda q: q @ P0)(Q)

        # 6) Stress errors
        P_err = jnp.linalg.norm(P_rot - P_expected, axis=(2, 3))

        return {
            "W_error_mean": None,  # no energy for FFNN_F
            "W_error_max":  None,
            "P_error_mean": float(jnp.mean(P_err)),
            "P_error_max":  float(jnp.max(P_err)),
        }

    # ----------------------------------------------------------
    # Unknown model_type
    # ----------------------------------------------------------
    raise ValueError(f"Unknown model_type '{model_type}'.")

#helper fct to get model predictions for eval

def predict_P_from_F(model, model_type: str, F, G=None):
    """
    Compute model stress predictions P(F) for different model types.

    Parameters
    ----------
    model : trained model
    model_type : {"WF", "WI", "WI_Cubic", "FFNN_F"}
    F : (N, 3, 3) JAX array
    G : structural tensor, required for "WI_Cubic" (and "WI" if you use cubic invariants there)

    Returns
    -------
    P_pred : (N, 3, 3) JAX array
    """
    #Defining Structural Tensors
    # Transversly isotropic
    G_ti = jnp.array([[4.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5]])

    # Cubic
    G_cub = td2.G_cub()

    model_type = model_type.upper()

    if model_type == "WF":
        # Deformation-based PANN: model(F) -> (W, P)
        W_pred, P_pred = jax.vmap(model)(F)

    elif model_type == "WI":
        # TI invariant-based PANN; assume batch helper exists
        I = td2.compute_all_invariants(F, G)  # (N, dim_I)
        W_pred, P_pred = jax.vmap(model)((F, I))

    elif model_type == "WI_CUBIC":
        assert G is not None, "G tensor required for WI_Cubic"
        I = td2.compute_all_invariants_cubic(F, G)  # (N, dim_I)
        W_pred, P_pred = jax.vmap(model)((F, I))

    elif model_type == "FFNN_F":
        # Naive FFNN trained on C (6 independent components) but we start from F.
        # 1) F -> C = F^T F
        def F_to_C6(F_single):
            C = F_single.T @ F_single
            # Voigt-like 6-vector: [C11, C22, C33, C12, C13, C23]
            return jnp.array([
                C[0, 0],
                C[1, 1],
                C[2, 2],
                C[0, 1],
                C[0, 2],
                C[1, 2],
            ])

        C6 = jax.vmap(F_to_C6)(F)          # (N, 6)

        # 2) Feed C6 into the FFNN
        P_raw = jax.vmap(model)(C6)        # (N,9) or (N,3,3)

        # 3) Reshape to (N,3,3)
        if P_raw.ndim == 2 and P_raw.shape[1] == 9:
            P_pred = P_raw.reshape(-1, 3, 3)
        elif P_raw.ndim == 3 and P_raw.shape[1:] == (3, 3):
            P_pred = P_raw
        else:
            raise ValueError(
                f"FFNN_F expected output shape (N,9) or (N,3,3), got {P_raw.shape}"
            )

    else:
        raise ValueError(f"Unknown model_type '{model_type}'.")

    return P_pred


# 3x3 true - pred plot for each P component
def parity_plot_P(
    model,
    model_type: str,
    F,
    P_true,
    G=None,
    title: str = None,
):
    """
    Parity plots (predicted vs. true) for all 9 components of P.

    Parameters
    ----------
    model, model_type : see predict_P_from_F
    F : (N,3,3)
    P_true : (N,3,3)
    G : structural tensor if needed
    """
    P_pred = predict_P_from_F(model, model_type, F, G=G)

    P_true = np.array(P_true)
    P_pred = np.array(P_pred)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(title or f"Parity plots for P – {model_type}", fontsize=14)

    comp_labels = [["11", "12", "13"],
                   ["21", "22", "23"],
                   ["31", "32", "33"]]

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            x = P_true[:, i, j]
            y = P_pred[:, i, j]

            ax.scatter(x, y, s=10, alpha=0.4)
            # diagonal
            mn = min(x.min(), y.min())
            mx = max(x.max(), y.max())
            ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)

            ax.set_xlabel(f"P_true{comp_labels[i][j]}")
            ax.set_ylabel(f"P_pred{comp_labels[i][j]}")
            ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

#single model eval of error distribution
def error_histograms_P(
    model,
    model_type: str,
    F,
    P_true,
    G=None,
    bins: int = 30,
    title: str = None,
):
    """
    Histograms of per-component errors P_pred - P_true.

    Parameters
    ----------
    model, model_type : see predict_P_from_F
    F : (N,3,3)
    P_true : (N,3,3)
    """
    P_pred = predict_P_from_F(model, model_type, F, G=G)

    P_true = np.array(P_true)
    P_pred = np.array(P_pred)

    errors = P_pred - P_true  # (N,3,3)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(title or f"Error histograms for P – {model_type}", fontsize=14)

    comp_labels = [["11", "12", "13"],
                   ["21", "22", "23"],
                   ["31", "32", "33"]]

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            e = errors[:, i, j]
            ax.hist(e, bins=bins, density=True, alpha=0.7)
            ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
            ax.set_xlabel(f"Error P{comp_labels[i][j]}")
            ax.set_ylabel("Density")
            ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

#tbd
def error_vs_magnitude_P(
    model,
    model_type: str,
    F,
    P_true,
    G=None,
    title: str = None,
):
    """
    Plot |error_ij| vs ||P_true||_F for each component.

    Parameters
    ----------
    model, model_type : see predict_P_from_F
    F : (N,3,3)
    P_true : (N,3,3)
    """
    P_pred = predict_P_from_F(model, model_type, F, G=G)

    P_true = np.array(P_true)
    P_pred = np.array(P_pred)

    errors = np.abs(P_pred - P_true)      # (N,3,3)
    mag = np.linalg.norm(P_true, axis=(1, 2))  # (N,)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(title or f"|error| vs ||P_true||_F – {model_type}", fontsize=14)

    comp_labels = [["11", "12", "13"],
                   ["21", "22", "23"],
                   ["31", "32", "33"]]

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            e = errors[:, i, j]
            ax.scatter(mag, e, s=10, alpha=0.4)
            ax.set_xlabel(r"$\|P_\mathrm{true}\|_F$")
            ax.set_ylabel(f"|error P{comp_labels[i][j]}|")
            ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

#multiple model comparison with 3x3 error in P components through boxplots
def _to_stacked_preds(P_pred):
    """
    Normalize P_pred into shape (K, N, 3, 3).
    Accepts:
      - (N,3,3)
      - (K,N,3,3)
      - list/tuple of (N,3,3)
    """
    if isinstance(P_pred, (list, tuple)):
        arr = np.stack([np.array(x) for x in P_pred], axis=0)  # (K,N,3,3)
    else:
        arr = np.array(P_pred)
        if arr.ndim == 3 and arr.shape[-2:] == (3, 3):
            arr = arr[None, ...]  # (1,N,3,3)
        elif arr.ndim == 4 and arr.shape[-2:] == (3, 3):
            pass  # already (K,N,3,3)
        else:
            raise ValueError(f"Unsupported prediction shape: {arr.shape}")
    return arr


def component_error_distribution_grid(
    P_true,
    P_pred_dict: dict,
    title: str = "Per-component error distributions (boxplots)",
    init_reduce: str = "mean",
    use_symlog: bool = False,
    symlog_linthresh: float = 1e-3,
):
    """
    3x3 grid; each subplot is one P_ij component.
    Uses shared y-axis limits across all subplots.
    """
    P_true = np.array(P_true)

    model_names = list(P_pred_dict.keys())
    display_names = [short_run_label(n, multiline=True) for n in model_names]
    num_models = len(model_names)

    # -------------------------------------------------
    # 1) Compute global y-limits (shared across all axes)
    # -------------------------------------------------
    all_errors = []
    for name in model_names:
        preds = _to_stacked_preds(P_pred_dict[name])  # (K,N,3,3)

        if init_reduce == "mean":
            P_pred_red = preds.mean(axis=0)
        else:
            raise ValueError(f"Unknown init_reduce='{init_reduce}'")

        err = P_pred_red - P_true  # (N,3,3)
        all_errors.append(err.reshape(-1))

    all_errors = np.concatenate(all_errors)
    e_min, e_max = float(all_errors.min()), float(all_errors.max())
    pad = 0.05 * max(abs(e_min), abs(e_max), 1e-12)
    y_min, y_max = e_min - pad, e_max + pad

    # -------------------------------------------------
    # 2) Plot grid with shared y-limits
    # -------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharey=True)
    fig.suptitle(title, fontsize=14)

    comp_labels = [["11", "12", "13"],
                   ["21", "22", "23"],
                   ["31", "32", "33"]]

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            data = []

            for name in model_names:
                preds = _to_stacked_preds(P_pred_dict[name])  # (K,N,3,3)
                if init_reduce == "mean":
                    P_pred_red = preds.mean(axis=0)
                else:
                    raise ValueError(f"Unknown init_reduce='{init_reduce}'")

                e = (P_pred_red - P_true)[:, i, j]  # (N,)
                data.append(e)

            bp = ax.boxplot(data, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_alpha(0.6)

            ax.set_ylim(y_min, y_max)
            if use_symlog:
                ax.set_yscale("symlog", linthresh=symlog_linthresh)

            ax.set_xticks(range(1, num_models + 1))
            ax.set_xticklabels(display_names, rotation=0, ha="center", fontsize=7)
            ax.set_ylabel(f"Error P{comp_labels[i][j]}")
            ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def component_rmse_barplot(
    P_true,
    P_pred_dict: dict,
    title: str = "Per-component RMSE of P",
    init_reduce: str = "mean",
):
    """
    Grouped bar plot: RMSE per component for each model.
    """
    P_true = np.array(P_true)
    model_names = list(P_pred_dict.keys())
    legend_names = [short_run_label(n, multiline=False) for n in model_names]
    num_models = len(model_names)

    comp_labels_flat = ["11", "12", "13", "21", "22", "23", "31", "32", "33"]
    x = np.arange(len(comp_labels_flat))
    width = 0.8 / max(num_models, 1)

    rmse = {}
    for name in model_names:
        preds = _to_stacked_preds(P_pred_dict[name])  # (K,N,3,3)
        if init_reduce == "mean":
            P_pred_red = preds.mean(axis=0)
        else:
            raise ValueError(f"Unknown init_reduce='{init_reduce}'")

        e = P_pred_red - P_true  # (N,3,3)
        rmse[name] = np.sqrt(np.mean(e**2, axis=0)).reshape(-1)  # (9,)

    fig, ax = plt.subplots(figsize=(12.5, 5.2))

    for idx, (name, leg) in enumerate(zip(model_names, legend_names)):
        offset = (idx - (num_models - 1) / 2) * width
        ax.bar(x + offset, rmse[name], width=width, label=leg, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"P{c}" for c in comp_labels_flat])
    ax.set_ylabel("RMSE")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Legend outside to avoid covering bars
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8,
    )

    # Reserve space on the right for the legend
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    plt.show()


def _stack_preds(preds):
    """
    Normalize predictions to shape (K, N, ...)

    Accepts:
      - array (N, ...)
      - array (K, N, ...)
      - list of arrays [(N, ...), ...]

    Returns:
      array (K, N, ...)
    """
    if isinstance(preds, (list, tuple)):
        arr = np.stack([np.array(p) for p in preds], axis=0)
    else:
        arr = np.array(preds)
        if arr.ndim >= 1:
            arr = arr[None, ...]
        else:
            raise ValueError(f"Unsupported prediction shape: {arr.shape}")
    return arr

def rmse_energy_and_stress_barplots(
    W_true,
    P_true,
    W_pred_dict: dict,
    P_pred_dict: dict,
    *,
    title_energy: str = "Energy RMSE",
    title_stress: str = "Stress RMSE",
):
    """
    Compute and plot RMSE for energy (W) and stress (P).
    """
    W_true = np.array(W_true).reshape(-1)
    P_true = np.array(P_true)

    model_names = list(W_pred_dict.keys())
    xlabels = [short_run_label(n, multiline=True) for n in model_names]

    # ---- RMSE(W) ----
    rmse_W = {}
    for name in model_names:
        W_preds = _stack_preds(W_pred_dict[name])  # (K,N) or (K,N,1)
        W_mean = np.squeeze(W_preds.mean(axis=0))
        err = W_mean - W_true
        rmse_W[name] = float(np.sqrt(np.mean(err**2)))

    # ---- RMSE(P) ----
    rmse_P = {}
    for name in model_names:
        P_preds = _stack_preds(P_pred_dict[name])  # (K,N,3,3)
        P_mean = P_preds.mean(axis=0)
        err = P_mean - P_true
        rmse_P[name] = float(np.sqrt(np.mean(err**2)))

    # -----------------
    # Plot: Energy RMSE
    # -----------------
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(range(len(model_names)), [rmse_W[n] for n in model_names], alpha=0.85)
    ax.set_ylabel("RMSE (Energy W)")
    ax.set_title(title_energy)
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(xlabels, fontsize=7)
    fig.tight_layout()
    plt.show()

    # -----------------
    # Plot: Stress RMSE
    # -----------------
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(range(len(model_names)), [rmse_P[n] for n in model_names], alpha=0.85)
    ax.set_ylabel("RMSE (Stress P)")
    ax.set_yscale("log")
    ax.set_title(title_stress)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(xlabels, fontsize=7)
    fig.tight_layout()
    plt.show()

@dataclass(frozen=True)
class RMSEReport:
    """
    A structured return type so you can keep notebooks clean.
    """
    model_name: str
    model_id: str
    test_mode: str
    n_inits: int

    # Stress metrics
    rmse_P_per_init: np.ndarray          # (K,)
    rmse_P_mean: float
    rmse_P_median: float                # NEW
    rmse_P_std: float

    rmse_P_comp_per_init: np.ndarray     # (K, 3, 3)
    rmse_P_comp_mean: np.ndarray         # (3, 3)
    rmse_P_comp_std: np.ndarray          # (3, 3)

    bias_P_comp_per_init: np.ndarray     # (K, 3, 3)
    bias_P_comp_mean: np.ndarray         # (3, 3)
    bias_P_comp_std: np.ndarray          # (3, 3)

    # Energy metrics (optional; None if not available)
    rmse_W_per_init: np.ndarray | None   # (K,) or None
    rmse_W_mean: float | None
    rmse_W_median: float | None          # NEW
    rmse_W_std: float | None

    # Optional raw error tensors
    errors_P: np.ndarray | None          # (K, N, 3, 3) or None
    errors_W: np.ndarray | None          # (K, N) or None

def _rmse_scalar(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))

def _as_np(x) -> np.ndarray:
    return np.array(x)

def _parse_test_set_any(ts: Any, test_key: str):
    """
    Parse a test set from wf.get_test_data_for_run(...) (via ewf.get_test_sets).
    Supports:
      - MS/MSW:         (X, Y) where Y is (N,9) or (N,3,3)
      - WITI/WICUB:     ((F, I), (W, P))
      - WF/WF_AUG:      (F, (W, P))   OR sometimes ((F, I), (W, P)) depending on your pipeline

    Returns:
      inputs: tuple to feed into prediction dispatcher
      W_true: (N,) or None
      P_true: (N,3,3)
    """
    item = ts[test_key]

    # Case A: invariant-style: ((F,I),(W,P))
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
        (F, I), (W, P) = item
        W_true = jnp.squeeze(W)
        P_true = P
        return (F, I), W_true, P_true

    # Case B: deformation-style: (F, (W,P))
    if isinstance(item, tuple) and len(item) == 2 and not isinstance(item[0], tuple):
        F = item[0]
        target = item[1]
        if isinstance(target, tuple) and len(target) == 2:
            W, P = target
            return (F,), jnp.squeeze(W), P
        # fallback: stress-only target
        Y = target
        # Y could be (N,9) or (N,3,3)
        Y = jnp.array(Y)
        if Y.ndim == 2 and Y.shape[1] == 9:
            P_true = Y.reshape(-1, 3, 3)
        elif Y.ndim == 3 and Y.shape[-2:] == (3, 3):
            P_true = Y
        else:
            raise ValueError(f"Unsupported target shape for test set '{test_key}': {Y.shape}")
        return (F,), None, P_true

    raise ValueError(f"Unrecognized test-set structure for key='{test_key}': {type(item)} / {item}")

def _get_test_mode_keys(test_mode: str) -> list[str]:
    """
    test_mode:
      - "biax"
      - "mixed"
      - "full" (biax + mixed, concatenated consistently)
    """
    tm = test_mode.lower()
    if tm == "biax":
        return ["biax_test"]
    if tm == "mixed":
        return ["mixed_test"]
    if tm == "full":
        return ["biax_test", "mixed_test"]
    raise ValueError("test_mode must be one of {'biax','mixed','full'}")

def _concat_tests(parsed_list):
    """
    parsed_list: list of (inputs_tuple, W_true_or_None, P_true)
    Concats along N.
    Works for both:
      - inputs=(X,) or (F,) or (F,I)
    """
    inputs0, W0, P0 = parsed_list[0]

    # concat inputs
    if len(inputs0) == 1:
        Xs = [p[0][0] for p in parsed_list]
        inputs = (jnp.concatenate(Xs, axis=0),)
    elif len(inputs0) == 2:
        Fs = [p[0][0] for p in parsed_list]
        Is = [p[0][1] for p in parsed_list]
        inputs = (jnp.concatenate(Fs, axis=0), jnp.concatenate(Is, axis=0))
    else:
        raise ValueError(f"Unsupported input tuple length: {len(inputs0)}")

    # concat W if available in all parts
    if all(p[1] is not None for p in parsed_list):
        Ws = [p[1] for p in parsed_list]
        W_true = jnp.concatenate(Ws, axis=0)
    else:
        W_true = None

    Ps = [p[2] for p in parsed_list]
    P_true = jnp.concatenate(Ps, axis=0)

    return inputs, W_true, P_true

def _predict_WP(model: Any, inputs: tuple[Any, ...]):
    """
    Robust prediction wrapper.
    Returns:
      W_pred_or_None: (N,) or None
      P_pred: (N,3,3)
    """
    # Stress-only models (MS/MSW etc.) are usually called model(X)->(N,9)
    if len(inputs) == 1:
        X = inputs[0]
        try:
            # Try stress-only output first: (N,9) or (N,3,3)
            Y = jax.vmap(model)(X)
            Y = jnp.array(Y)
            if Y.ndim == 2 and Y.shape[1] == 9:
                return None, Y.reshape(-1, 3, 3)
            if Y.ndim == 3 and Y.shape[-2:] == (3, 3):
                return None, Y
            # Otherwise it might be (W,P) tuple per sample: model(F)->(W,P)
        except Exception:
            pass

        # Try energy+stress signature: model(F)->(W,P)
        W, P = jax.vmap(model)(X)
        return jnp.squeeze(W), P

    # Invariant-based: model((F,I))->(W,P)
    if len(inputs) == 2:
        F, I = inputs
        W, P = jax.vmap(model)((F, I))
        return jnp.squeeze(W), P

    raise ValueError(f"Unsupported inputs tuple length: {len(inputs)}")

def compute_rmse_over_test_set(
    runs: Iterable[Any],
    *,
    dataset_1: dict | None = None,
    G_cub: jnp.ndarray | None = None,
    test_mode: str = "full",
    model_name: str | None = None,
    return_component_metrics: bool = True,
    return_raw_errors: bool = False,
) -> RMSEReport:
    runs = list(runs)
    if not runs:
        raise ValueError("compute_rmse_over_test_set received an empty runs iterable.")

    r0 = runs[0]
    mid = str(getattr(r0, "model_id", "")).upper()
    if model_name is None:
        model_name = getattr(r0, "base_tag", None) or getattr(r0, "tag", "model")

    ts = ewf.get_test_sets(r0, dataset_1=dataset_1, G_cub=G_cub)
    keys = _get_test_mode_keys(test_mode)

    parsed = [_parse_test_set_any(ts, k) for k in keys]
    if len(parsed) == 1:
        inputs, W_true, P_true = parsed[0]
    else:
        inputs, W_true, P_true = _concat_tests(parsed)

    P_true_np = _as_np(P_true)  # (N,3,3)
    W_true_np = _as_np(W_true) if W_true is not None else None

    # --- Predict per init ---
    P_errs = []
    W_errs = []

    for r in runs:
        W_pred, P_pred = _predict_WP(r.model, inputs)
        P_pred_np = _as_np(P_pred)

        P_err = P_pred_np - P_true_np   # (N,3,3)
        P_errs.append(P_err)

        if W_true_np is not None and W_pred is not None:
            W_pred_np = np.squeeze(_as_np(W_pred))
            W_errs.append(W_pred_np - np.squeeze(W_true_np))

    P_errs = np.stack(P_errs, axis=0)  # (K,N,3,3)
    K = P_errs.shape[0]

    # --- Global stress RMSE per init ---
    rmse_P_per_init = np.sqrt(np.mean(P_errs**2, axis=(1, 2, 3)))  # (K,)
    rmse_P_mean   = float(np.mean(rmse_P_per_init))
    rmse_P_median = float(np.median(rmse_P_per_init))              # NEW
    rmse_P_std    = float(np.std(rmse_P_per_init))

    # --- Component metrics ---
    if return_component_metrics:
        rmse_P_comp_per_init = np.sqrt(np.mean(P_errs**2, axis=1))      # (K,3,3)
        rmse_P_comp_mean = np.mean(rmse_P_comp_per_init, axis=0)        # (3,3)
        rmse_P_comp_std  = np.std(rmse_P_comp_per_init, axis=0)         # (3,3)

        bias_P_comp_per_init = np.mean(P_errs, axis=1)                  # (K,3,3)
        bias_P_comp_mean = np.mean(bias_P_comp_per_init, axis=0)        # (3,3)
        bias_P_comp_std  = np.std(bias_P_comp_per_init, axis=0)         # (3,3)
    else:
        rmse_P_comp_per_init = np.zeros((K, 3, 3))
        rmse_P_comp_mean = np.zeros((3, 3))
        rmse_P_comp_std = np.zeros((3, 3))
        bias_P_comp_per_init = np.zeros((K, 3, 3))
        bias_P_comp_mean = np.zeros((3, 3))
        bias_P_comp_std = np.zeros((3, 3))

    # --- Energy RMSE (optional) ---
    if W_true_np is not None and len(W_errs) == K:
        W_errs = np.stack(W_errs, axis=0)  # (K,N)
        rmse_W_per_init = np.sqrt(np.mean(W_errs**2, axis=1))  # (K,)
        rmse_W_mean   = float(np.mean(rmse_W_per_init))
        rmse_W_median = float(np.median(rmse_W_per_init))      # NEW
        rmse_W_std    = float(np.std(rmse_W_per_init))
        errors_W_out = W_errs if return_raw_errors else None
    else:
        rmse_W_per_init = None
        rmse_W_mean = None
        rmse_W_median = None   # NEW
        rmse_W_std = None
        errors_W_out = None

    errors_P_out = P_errs if return_raw_errors else None

    return RMSEReport(
        model_name=str(model_name),
        model_id=mid,
        test_mode=test_mode,
        n_inits=K,
        rmse_P_per_init=rmse_P_per_init,
        rmse_P_mean=rmse_P_mean,
        rmse_P_median=rmse_P_median,     # NEW
        rmse_P_std=rmse_P_std,
        rmse_P_comp_per_init=rmse_P_comp_per_init,
        rmse_P_comp_mean=rmse_P_comp_mean,
        rmse_P_comp_std=rmse_P_comp_std,
        bias_P_comp_per_init=bias_P_comp_per_init,
        bias_P_comp_mean=bias_P_comp_mean,
        bias_P_comp_std=bias_P_comp_std,
        rmse_W_per_init=rmse_W_per_init,
        rmse_W_mean=rmse_W_mean,
        rmse_W_median=rmse_W_median,     # NEW
        rmse_W_std=rmse_W_std,
        errors_P=errors_P_out,
        errors_W=errors_W_out,
    )

def select_best_per_size(
    reports: dict,
    *,
    criterion: str = "median",   # "median" or "mean"
    metric: str = "P",           # "P" (stress) or "W" (energy)
    sizes=("small", "medium", "large"),
) -> dict:
    """
    From a dict[name -> RMSEReport] that contains multiple variants per size
    (e.g., steps sweeps), return dict[size -> (name, report)] for the best one.
    """
    criterion = criterion.lower().strip()
    metric = metric.upper().strip()
    if criterion not in ("median", "mean"):
        raise ValueError("criterion must be 'median' or 'mean'.")
    if metric not in ("P", "W"):
        raise ValueError("metric must be 'P' or 'W'.")

    def score(rep):
        if metric == "P":
            return rep.rmse_P_median if criterion == "median" else rep.rmse_P_mean
        else:
            # Energy might be None for some runs
            val = rep.rmse_W_median if criterion == "median" else rep.rmse_W_mean
            return np.inf if val is None else float(val)

    winners = {}
    for size in sizes:
        # filter all entries belonging to this size
        candidates = [(name, rep) for name, rep in reports.items() if size in name]
        if not candidates:
            continue
        winners[size] = min(candidates, key=lambda nr: score(nr[1]))  # (name, report)

    return winners

def plot_best_sizes_mean_median(
    winners: dict,
    *,
    metric: str = "P",
    title: str | None = None,
    logy: bool = True,
    figsize=(8.5, 4.8),
):
    """
    winners: dict[size -> (name, RMSEReport)] as returned by select_best_per_size().
    """
    metric = metric.upper().strip()
    if metric not in ("P", "W"):
        raise ValueError("metric must be 'P' or 'W'.")

    # keep consistent ordering if present
    order = [s for s in ("small", "medium", "large") if s in winners]
    labels = []
    means = []
    meds = []

    for s in order:
        name, rep = winners[s]
        if metric == "P":
            means.append(rep.rmse_P_mean)
            meds.append(rep.rmse_P_median)
            ylabel = "RMSE (Stress P)"
            if title is None:
                title = "Best per architecture: Stress RMSE mean vs median (over inits)"
        else:
            means.append(np.nan if rep.rmse_W_mean is None else rep.rmse_W_mean)
            meds.append(np.nan if rep.rmse_W_median is None else rep.rmse_W_median)
            ylabel = "RMSE (Energy W)"
            if title is None:
                title = "Best per architecture: Energy RMSE mean vs median (over inits)"

        # label includes which variant won (e.g., steps)
        labels.append(f"{s}\n{short_run_label(name, multiline=True)}")

    x = np.arange(len(order))
    width = 0.38

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width/2, means, width=width, label="mean", alpha=0.85)
    ax.bar(x + width/2, meds,  width=width, label="median", alpha=0.85)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    plt.show()

def _steps_to_k(steps: int) -> str:
    if steps >= 1_000_000:
        val = steps / 1_000_000
        return f"{val:g}M"
    if steps >= 1_000:
        return f"{steps//1000}k"
    return str(steps)

def _arch_to_size(l: int, n: int) -> str:
    if (l, n) == (2, 8):
        return "S"
    if (l, n) == (3, 16):
        return "M"
    if (l, n) == (4, 32):
        return "L"
    return f"l{l}n{n}"

def _size_word_to_letter(size_word: str) -> str:
    size_word = size_word.lower()
    if size_word == "small":
        return "S"
    if size_word == "medium":
        return "M"
    if size_word == "large":
        return "L"
    return "?"

def short_run_label(name: str, *, multiline: bool = True) -> str:
    """
    Robust compact label builder across tasks.

    Supports:
      - Full tags with arch+steps:   ..._l{l}_n{n}_steps{steps}...
      - Task 2.2 synthetic keys:     small_steps100000, medium_steps300000, ...
      - Drops init aggregation suffixes: _avg20inits -> avg
    """
    # Normalize avg suffix
    s = re.sub(r"_avg\d+inits$", "_avg", name)
    s = re.sub(r"_avg\d+$", "_avg", s)

    # ---------- Case A: Task 2.2 synthetic keys like "small_steps100000" ----------
    m = re.match(r"^(small|medium|large)_steps(\d+)$", s, flags=re.IGNORECASE)
    if m:
        size_letter = _size_word_to_letter(m.group(1))
        steps_str = _steps_to_k(int(m.group(2)))
        model_id = "MS"  # your Task 2.2 model family
        avg_flag = "avg"
        parts = [model_id, size_letter, steps_str, avg_flag]
        return "\n".join(parts) if multiline else " ".join(parts)

    # ---------- Case B: artifact tag like "MS_small_l2_n8_steps100000" ----------
    # model_id is first token before "_"
    model_id = s.split("_", 1)[0] if "_" in s else s

    # size from name if present (MS_small_ / MS_medium_ / MS_large_)
    msize = re.search(r"_(small|medium|large)_", s, flags=re.IGNORECASE)
    size_letter_from_word = _size_word_to_letter(msize.group(1)) if msize else None

    # arch+steps if present
    march = re.search(r"_l(\d+)_n(\d+)_steps(\d+)", s)
    if march:
        l = int(march.group(1))
        n = int(march.group(2))
        steps = int(march.group(3))
        # prefer S/M/L mapping via (l,n); fallback to size word if available
        size_letter = _arch_to_size(l, n)
        if size_letter.startswith("l") and size_letter_from_word is not None:
            size_letter = size_letter_from_word
        steps_str = _steps_to_k(steps)
    else:
        # fallback: if we at least have "..._steps123"
        msteps = re.search(r"_steps(\d+)", s)
        steps_str = _steps_to_k(int(msteps.group(1))) if msteps else "?"
        size_letter = size_letter_from_word if size_letter_from_word is not None else "?"

    avg_flag = "avg" if ("_avg" in s) else ""

    parts = [model_id, size_letter, steps_str]
    if avg_flag:
        parts.append(avg_flag)

    return "\n".join(parts) if multiline else " ".join(parts)

def compare_two_run_groups_mean_median_barplot(
    group_a_runs,
    group_b_runs,
    *,
    name_a: str,
    name_b: str,
    dataset_1=None,
    test_mode: str = "full",
    metric: str = "P",         # "P" (stress) or "W" (energy)
    logy: bool = True,
    title: str | None = None,
    figsize=(7.8, 4.6),
):
    """
    Compute RMSEReport for two groups (each group = list of inits) and plot mean vs median.

    Parameters
    ----------
    group_a_runs, group_b_runs:
        Iterables of Run objects (same model type per group), typically one per init.

    name_a, name_b:
        Labels used for the plot x-axis.

    dataset_1, test_mode:
        Passed to compute_rmse_over_test_set so both groups use consistent test logic.

    metric:
        "P" for stress RMSE, "W" for energy RMSE (if available).

    Returns
    -------
    (rep_a, rep_b)
    """
    rep_a = compute_rmse_over_test_set(
        list(group_a_runs),
        dataset_1=dataset_1,
        test_mode=test_mode,
        model_name=name_a,
    )
    rep_b = compute_rmse_over_test_set(
        list(group_b_runs),
        dataset_1=dataset_1,
        test_mode=test_mode,
        model_name=name_b,
    )

    metric = metric.upper().strip()
    if metric == "P":
        means = [rep_a.rmse_P_mean, rep_b.rmse_P_mean]
        meds  = [rep_a.rmse_P_median, rep_b.rmse_P_median]
        ylabel = "RMSE (Stress P)"
        if title is None:
            title = "Task 2.3 vs Task 2.2: Stress RMSE mean vs median (over inits)"
    elif metric == "W":
        means = [
            np.nan if rep_a.rmse_W_mean is None else rep_a.rmse_W_mean,
            np.nan if rep_b.rmse_W_mean is None else rep_b.rmse_W_mean,
        ]
        meds = [
            np.nan if rep_a.rmse_W_median is None else rep_a.rmse_W_median,
            np.nan if rep_b.rmse_W_median is None else rep_b.rmse_W_median,
        ]
        ylabel = "RMSE (Energy W)"
        if title is None:
            title = "Task 2.3 vs Task 2.2: Energy RMSE mean vs median (over inits)"
    else:
        raise ValueError("metric must be 'P' or 'W'.")

    # Plot
    xlabels = [short_run_label(name_a, multiline=True), short_run_label(name_b, multiline=True)]
    x = np.arange(2)
    width = 0.38

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width/2, means, width=width, label="mean", alpha=0.85)
    ax.bar(x + width/2, meds,  width=width, label="median", alpha=0.85)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)

    if logy:
        ax.set_yscale("log")

    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return rep_a, rep_b

def _arch_bucket_from_cfg_name(name: str) -> str:
    # name like "WITI_l3_n16_steps300000"
    m = re.search(r"_l(\d+)_n(\d+)_", name)
    if not m:
        return "other"
    l, n = int(m.group(1)), int(m.group(2))
    if (l, n) == (2, 8):
        return "small"
    if (l, n) == (3, 16):
        return "medium"
    if (l, n) == (4, 32):
        return "large"
    return f"l{l}_n{n}"

def select_best_per_arch_bucket(
    reports: dict[str, "RMSEReport"],
    *,
    metric: str = "P",          # "P" or "W"
    criterion: str = "median",  # "median" or "mean"
) -> dict[str, tuple[str, "RMSEReport"]]:
    metric = metric.upper().strip()
    criterion = criterion.lower().strip()
    assert metric in ("P", "W")
    assert criterion in ("median", "mean")

    def score(rep):
        if metric == "P":
            return rep.rmse_P_median if criterion == "median" else rep.rmse_P_mean
        else:
            v = rep.rmse_W_median if criterion == "median" else rep.rmse_W_mean
            return float("inf") if v is None else float(v)

    buckets: dict[str, list[tuple[str, "RMSEReport"]]] = {}
    for name, rep in reports.items():
        b = _arch_bucket_from_cfg_name(name)
        buckets.setdefault(b, []).append((name, rep))

    winners = {}
    for b, items in buckets.items():
        if b == "other":
            continue
        winners[b] = min(items, key=lambda nr: score(nr[1]))  # (name, report)

    return winners

def get_train_witi_from_dataset_1(dataset_1: dict):
    """
    Reconstruct the exact calibration/training set used by Task 3 WITI workflows.
    Returns ((F_train, I_train), (W_train, P_train)).
    """
    F_train = jnp.concatenate([dataset_1["F_bi"],  dataset_1["F_uni"],  dataset_1["F_ps"]], axis=0)
    I_train = jnp.concatenate([dataset_1["I_bi"],  dataset_1["I_uni"],  dataset_1["I_ps"]], axis=0)
    W_train = jnp.concatenate([dataset_1["W_bi"],  dataset_1["W_uni"],  dataset_1["W_ps"]], axis=0)
    P_train = jnp.concatenate([dataset_1["P_bi"],  dataset_1["P_uni"],  dataset_1["P_ps"]], axis=0)
    return (F_train, I_train), (W_train, P_train)

def parity_plot_train_vs_test_witi(
    run, *,
    dataset_1: dict,
    test_which: str = "full",      # "mixed", "biax", "full"
    target: str = "W",             # "W" or "P"
    p_comp: tuple[int,int] = (0,0),# used if target="P"
    alpha: float = 0.35,
    s: float = 10.0,
    title: str | None = None,
    figsize=(6.5, 6.0),
):
    """
    Parity plot: x=true, y=pred. Training points vs test points in different colors.
    """
    # --- train set (calibration) ---
    (F_tr, I_tr), (W_tr, P_tr) = get_train_witi_from_dataset_1(dataset_1)

    # --- test set ---
    (F_te, I_te), (W_te, P_te) = ewf.get_test_witi(run, dataset_1=dataset_1, which=test_which)

    # --- predict ---
    Wp_tr, Pp_tr = jax.vmap(run.model)((F_tr, I_tr))
    Wp_te, Pp_te = jax.vmap(run.model)((F_te, I_te))

    Wp_tr = jnp.squeeze(Wp_tr); Wp_te = jnp.squeeze(Wp_te)

    # --- select scalar to plot ---
    if target.upper() == "W":
        x_tr = np.asarray(W_tr).reshape(-1)
        y_tr = np.asarray(Wp_tr).reshape(-1)
        x_te = np.asarray(W_te).reshape(-1)
        y_te = np.asarray(Wp_te).reshape(-1)
        ylabel = "Predicted W"
        xlabel = "True W"
        if title is None:
            title = f"WITI parity (W): train vs test ({test_which})"
    else:
        i, j = p_comp
        x_tr = np.asarray(P_tr[:, i, j]).reshape(-1)
        y_tr = np.asarray(Pp_tr[:, i, j]).reshape(-1)
        x_te = np.asarray(P_te[:, i, j]).reshape(-1)
        y_te = np.asarray(Pp_te[:, i, j]).reshape(-1)
        ylabel = f"Predicted P[{i+1}{j+1}]"
        xlabel = f"True P[{i+1}{j+1}]"
        if title is None:
            title = f"WITI parity (P[{i+1}{j+1}]): train vs test ({test_which})"

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x_tr, y_tr, s=s, alpha=alpha, label="train (calibration)")
    ax.scatter(x_te, y_te, s=s, alpha=alpha, label="test")

    # y=x reference
    lo = min(np.min(x_tr), np.min(x_te), np.min(y_tr), np.min(y_te))
    hi = max(np.max(x_tr), np.max(x_te), np.max(y_tr), np.max(y_te))
    ax.plot([lo, hi], [lo, hi], linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    plt.show()

def pick_median_init_run(
    runs: list,
    *,
    dataset_1: dict,
    test_mode: str = "full",
) -> "Run":
    """
    Pick the run whose stress-RMSE is the median among inits (on the chosen test set).
    """
    rep = eval.compute_rmse_over_test_set(
        runs,
        dataset_1=dataset_1,
        test_mode=test_mode,
        model_name="tmp",
    )
    rmse = np.asarray(rep.rmse_P_per_init)
    k = int(np.argsort(rmse)[len(rmse)//2])
    return list(runs)[k]

def get_train_witi_from_dataset_1(dataset_1):
    F_tr = jnp.concatenate(
        [dataset_1["F_bi"], dataset_1["F_uni"], dataset_1["F_ps"]], axis=0
    )
    I_tr = jnp.concatenate(
        [dataset_1["I_bi"], dataset_1["I_uni"], dataset_1["I_ps"]], axis=0
    )
    W_tr = jnp.concatenate(
        [dataset_1["W_bi"], dataset_1["W_uni"], dataset_1["W_ps"]], axis=0
    )
    P_tr = jnp.concatenate(
        [dataset_1["P_bi"], dataset_1["P_uni"], dataset_1["P_ps"]], axis=0
    )
    return (F_tr, I_tr), (W_tr, P_tr)

def plot_task3_section2_train_test_rmse_vs_steps(
    *,
    art_dir: str,
    dataset_1: dict,
    metric: str = "P",                 # "P" or "W"
    agg: str = "median",               # "median" or "mean" over inits
    figsize=(9.0, 8.5),
):
    """
    Task 3 Section 2 diagnostic plot (Option A):
      1) Training RMSE vs steps
      2) Test RMSE (biax) vs steps
      3) Test RMSE (mixed) vs steps

    - 3 panels share x-axis
    - log y-scale
    - SAME y-limits across all panels for visual comparability
    - red dotted horizontal line at the minimum RMSE in each panel
    """

    metric = metric.upper().strip()
    if metric not in ("P", "W"):
        raise ValueError("metric must be 'P' or 'W'.")

    agg = agg.lower().strip()
    if agg not in ("median", "mean"):
        raise ValueError("agg must be 'median' or 'mean'.")

    # ----------------------------
    # Load runs
    # ----------------------------
    runs = ewf.load_runs(art_dir, model_id="WITI", dataset_1=dataset_1, strict=False)
    if not runs:
        raise FileNotFoundError(f"No WITI runs found in {art_dir}")

    # ----------------------------
    # Group runs by (l, n, steps)
    # ----------------------------
    def cfg_key(r):
        m = re.search(r"_l(\d+)_n(\d+)_steps(\d+)", r.tag)
        if not m:
            return None
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    groups = ewf.group_runs(runs, by=cfg_key)
    groups.pop(None, None)

    # ----------------------------
    # Architecture labels (by (l,n))
    # ----------------------------
    arch_map = {(2, 8): "S", (3, 16): "M", (4, 32): "L"}
    arch_order = ["S", "M", "L"]
    colors = {"S": "C0", "M": "C1", "L": "C2"}

    # ----------------------------
    # Build train set + both test sets once
    # ----------------------------
    (F_tr, I_tr), (W_tr, P_tr) = get_train_witi_from_dataset_1(dataset_1)

    any_run = runs[0]
    (F_bx, I_bx), (W_bx, P_bx) = ewf.get_test_witi(any_run, dataset_1=dataset_1, which="biax")
    (F_mx, I_mx), (W_mx, P_mx) = ewf.get_test_witi(any_run, dataset_1=dataset_1, which="mixed")

    # ----------------------------
    # Compute RMSE curves: per arch, per steps, per panel
    # ----------------------------
    # data[arch]["steps"] = [...]
    # data[arch]["train"] = [...]
    # data[arch]["biax"]  = [...]
    # data[arch]["mixed"] = [...]
    data = {a: {"steps": [], "train": [], "biax": [], "mixed": []} for a in arch_order}

    def _rmse_from_preds(Wp, Pp, W_true, P_true):
        if metric == "P":
            err = np.asarray(Pp - P_true)  # (N,3,3)
            return float(np.sqrt(np.mean(err**2)))
        else:
            err = np.asarray(jnp.squeeze(Wp) - jnp.squeeze(W_true))  # (N,)
            return float(np.sqrt(np.mean(err**2)))

    def _aggregate(vals):
        vals = np.asarray(vals, dtype=float)
        return float(np.median(vals)) if agg == "median" else float(np.mean(vals))

    for (l, n, steps), cfg_runs in groups.items():
        arch = arch_map.get((l, n))
        if arch is None:
            continue

        rmse_tr_inits = []
        rmse_bx_inits = []
        rmse_mx_inits = []

        for r in cfg_runs:
            # train
            Wp_tr, Pp_tr = jax.vmap(r.model)((F_tr, I_tr))
            rmse_tr_inits.append(_rmse_from_preds(Wp_tr, Pp_tr, W_tr, P_tr))

            # test biax
            Wp_bx, Pp_bx = jax.vmap(r.model)((F_bx, I_bx))
            rmse_bx_inits.append(_rmse_from_preds(Wp_bx, Pp_bx, W_bx, P_bx))

            # test mixed
            Wp_mx, Pp_mx = jax.vmap(r.model)((F_mx, I_mx))
            rmse_mx_inits.append(_rmse_from_preds(Wp_mx, Pp_mx, W_mx, P_mx))

        data[arch]["steps"].append(int(steps))
        data[arch]["train"].append(_aggregate(rmse_tr_inits))
        data[arch]["biax"].append(_aggregate(rmse_bx_inits))
        data[arch]["mixed"].append(_aggregate(rmse_mx_inits))

    # ----------------------------
    # Determine common y-limits across ALL panels and curves
    # ----------------------------
    all_y = []
    for arch in arch_order:
        for k in ("train", "biax", "mixed"):
            all_y.extend(list(data[arch][k]))
    all_y = np.asarray(all_y, dtype=float)
    all_y = all_y[np.isfinite(all_y) & (all_y > 0)]
    if all_y.size == 0:
        raise RuntimeError("No RMSE values computed (check parsing/grouping).")

    # log-safe padding
    ymin = float(np.min(all_y) / 1.15)
    ymax = float(np.max(all_y) * 1.15)

    # panel-specific minima (for red dotted lines)
    def _panel_min(panel_key: str) -> float:
        vals = []
        for arch in arch_order:
            vals.extend(list(data[arch][panel_key]))
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        return float(np.min(vals))

    best_train = _panel_min("train")
    best_biax  = _panel_min("biax")
    best_mixed = _panel_min("mixed")

    # ----------------------------
    # Plot (3 aligned panels)
    # ----------------------------
    fig, (ax_tr, ax_bx, ax_mx) = plt.subplots(
        3, 1, sharex=True, figsize=figsize, gridspec_kw={"hspace": 0.05}
    )

    for arch in arch_order:
        steps = np.asarray(data[arch]["steps"], dtype=int)
        order = np.argsort(steps)
        steps_k = steps[order] / 1000.0

        y_tr = np.asarray(data[arch]["train"], dtype=float)[order]
        y_bx = np.asarray(data[arch]["biax"],  dtype=float)[order]
        y_mx = np.asarray(data[arch]["mixed"], dtype=float)[order]

        ax_tr.plot(steps_k, y_tr, marker="o", color=colors[arch], label=arch)
        ax_bx.plot(steps_k, y_bx, marker="o", color=colors[arch], label=arch)
        ax_mx.plot(steps_k, y_mx, marker="o", color=colors[arch], label=arch)

    # red dotted min lines per panel
    ax_tr.axhline(best_train, linestyle=":", color="red", linewidth=1.5)
    ax_bx.axhline(best_biax,  linestyle=":", color="red", linewidth=1.5)
    ax_mx.axhline(best_mixed, linestyle=":", color="red", linewidth=1.5)

    # common formatting
    title_metric = "P-RMSE" if metric == "P" else "W-RMSE"
    agg_txt = "median" if agg == "median" else "mean"

    ax_tr.set_title(f"Task 3 Section 2 — {title_metric} vs training steps ({agg_txt} over inits)")

    for ax, ylabel in zip(
        (ax_tr, ax_bx, ax_mx),
        ("Training RMSE", "Test RMSE (biax)", "Test RMSE (mixed)"),
    ):
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)   # keep shared scale (this part was good)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

        # keep dense ticks, but no special labels
        _apply_dense_log_ticks(ax)




    ax_mx.set_xlabel("Training steps [k]")

    # one legend (top panel)
    ax_tr.legend(title="Architecture")

    fig.tight_layout()
    plt.show()

def get_train_witi_from_dataset_1(dataset_1: dict):
    """
    Reconstruct the exact calibration/training set used by Task 3 WITI workflows.
    Returns ((F_train, I_train), (W_train, P_train)).
    """
    F_train = jnp.concatenate([dataset_1["F_bi"],  dataset_1["F_uni"],  dataset_1["F_ps"]], axis=0)
    I_train = jnp.concatenate([dataset_1["I_bi"],  dataset_1["I_uni"],  dataset_1["I_ps"]], axis=0)
    W_train = jnp.concatenate([dataset_1["W_bi"],  dataset_1["W_uni"],  dataset_1["W_ps"]], axis=0)
    P_train = jnp.concatenate([dataset_1["P_bi"],  dataset_1["P_uni"],  dataset_1["P_ps"]], axis=0)
    return (F_train, I_train), (W_train, P_train)

def _apply_dense_log_ticks(ax):
    """
    Make log axis show more tick labels and minor ticks (2..9 per decade).
    """
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Formatter to show labels like 2×10^-2 (instead of only 10^k)
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))
    ax.tick_params(axis="y", which="major", length=6)
    ax.tick_params(axis="y", which="minor", length=3)

def get_train_ms_from_dataset_1(dataset_1: dict):
    """
    Reconstruct the exact calibration/training set used by Task 2.2 MS workflows.

    Training order in workflow_task_2_2_train_ms_sweep:
      C_cal_MS = [C_uni, C_ps, C_bi]
      P_cal_MS = [P_uni, P_ps, P_bi]
    Then X = vmap(C_to_six)(C_cal_MS), Y = reshape(P_cal_MS)->(N,9)
    """
    C_cal = jnp.concatenate([dataset_1["C_uni"], dataset_1["C_ps"], dataset_1["C_bi"]], axis=0)
    P_cal = jnp.concatenate([dataset_1["P_uni"], dataset_1["P_ps"], dataset_1["P_bi"]], axis=0)

    X_train = jax.vmap(td2.C_to_six)(C_cal)                 # (N,6)
    Y_train = P_cal.reshape(P_cal.shape[0], 9)              # (N,9)
    return X_train, Y_train

def plot_task2_2_train_test_rmse_vs_steps(
    *,
    art_dir: str,
    dataset_1: dict,
    agg: str = "median",          # "median" or "mean" across inits
    figsize=(9.0, 8.0),
):
    """
    Task 2.2 diagnostic plot (3 panels):
      1) Training RMSE vs steps  (calibration set: uni + pure_shear + biax)
      2) Test RMSE (biax) vs steps
      3) Test RMSE (mixed) vs steps

    - architectures: S/M/L (from run.arch_name or tag prefix)
    - x-axis: steps [k]
    - log y-scale
    - same y-limits across panels
    - red dotted line at per-panel minimum (no labels)
    """

    agg = agg.lower().strip()
    if agg not in ("median", "mean"):
        raise ValueError("agg must be 'median' or 'mean'.")

    # ----------------------------
    # Load runs (MS only)
    # ----------------------------
    runs = ewf.load_runs(art_dir, model_id="MS", dataset_1=dataset_1, strict=False)
    if not runs:
        raise FileNotFoundError(f"No MS runs found in {art_dir}")

    # ----------------------------
    # Build train set and both test sets once
    # ----------------------------
    X_tr, Y_tr = get_train_ms_from_dataset_1(dataset_1)

    any_run = runs[0]
    X_bx, Y_bx = ewf.get_test_ms(any_run, dataset_1=dataset_1, which="biax")   # (N,6),(N,9)
    X_mx, Y_mx = ewf.get_test_ms(any_run, dataset_1=dataset_1, which="mixed")  # (N,6),(N,9)

    # ----------------------------
    # Group runs by (arch, steps)
    # ----------------------------
    def _arch_from_run(r):
        # prefer meta-derived field if present
        a = getattr(r, "arch_name", None)
        if a:
            return str(a).lower()
        # fallback: parse tag "MS_small_..."
        for s in ("small", "medium", "large"):
            if f"MS_{s}_" in r.tag:
                return s
        return None

    def cfg_key(r):
        arch = _arch_from_run(r)
        steps = int(getattr(r, "steps", -1) or -1)
        if steps < 0:
            m = re.search(r"_steps(\d+)", r.tag)
            steps = int(m.group(1)) if m else -1
        if arch is None or steps < 0:
            return None
        return (arch, steps)

    groups = ewf.group_runs(runs, by=cfg_key)
    groups.pop(None, None)

    arch_order = ["small", "medium", "large"]
    arch_label = {"small": "S", "medium": "M", "large": "L"}
    colors = {"small": "C0", "medium": "C1", "large": "C2"}

    # ----------------------------
    # Compute RMSE per (arch, steps), aggregated over inits
    # ----------------------------
    data = {a: {"steps": [], "train": [], "biax": [], "mixed": []} for a in arch_order}

    def _rmse(Y_pred, Y_true):
        err = np.asarray(Y_pred - Y_true)   # (N,9)
        return float(np.sqrt(np.mean(err**2)))

    def _aggregate(vals):
        vals = np.asarray(vals, dtype=float)
        return float(np.median(vals)) if agg == "median" else float(np.mean(vals))

    for (arch, steps), cfg_runs in groups.items():
        if arch not in data:
            continue

        rmse_tr_inits = []
        rmse_bx_inits = []
        rmse_mx_inits = []

        for r in cfg_runs:
            # fastest/consistent: call the model directly
            Yp_tr = jax.vmap(r.model)(X_tr)   # (N,9)
            Yp_bx = jax.vmap(r.model)(X_bx)
            Yp_mx = jax.vmap(r.model)(X_mx)

            rmse_tr_inits.append(_rmse(Yp_tr, Y_tr))
            rmse_bx_inits.append(_rmse(Yp_bx, Y_bx))
            rmse_mx_inits.append(_rmse(Yp_mx, Y_mx))

        data[arch]["steps"].append(int(steps))
        data[arch]["train"].append(_aggregate(rmse_tr_inits))
        data[arch]["biax"].append(_aggregate(rmse_bx_inits))
        data[arch]["mixed"].append(_aggregate(rmse_mx_inits))

    # ----------------------------
    # Shared y-limits across all panels
    # ----------------------------
    all_y = []
    for arch in arch_order:
        all_y.extend(data[arch]["train"])
        all_y.extend(data[arch]["biax"])
        all_y.extend(data[arch]["mixed"])

    all_y = np.asarray(all_y, dtype=float)
    all_y = all_y[np.isfinite(all_y) & (all_y > 0)]
    if all_y.size == 0:
        raise RuntimeError("No RMSE values computed. Check grouping/tag parsing.")

    ymin = float(np.min(all_y) / 1.15)
    ymax = float(np.max(all_y) * 1.15)

    def _panel_min(key):
        vals = []
        for arch in arch_order:
            vals.extend(data[arch][key])
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        return float(np.min(vals))

    best_train = _panel_min("train")
    best_biax  = _panel_min("biax")
    best_mixed = _panel_min("mixed")

    # ----------------------------
    # Plot: 3 aligned panels
    # ----------------------------
    fig, (ax_tr, ax_bx, ax_mx) = plt.subplots(
        3, 1, sharex=True, figsize=figsize, gridspec_kw={"hspace": 0.05}
    )

    for arch in arch_order:
        steps = np.asarray(data[arch]["steps"], dtype=int)
        order = np.argsort(steps)
        steps_k = steps[order] / 1000.0

        y_tr = np.asarray(data[arch]["train"], dtype=float)[order]
        y_bx = np.asarray(data[arch]["biax"], dtype=float)[order]
        y_mx = np.asarray(data[arch]["mixed"], dtype=float)[order]

        ax_tr.plot(steps_k, y_tr, marker="o", color=colors[arch], label=arch_label[arch])
        ax_bx.plot(steps_k, y_bx, marker="o", color=colors[arch], label=arch_label[arch])
        ax_mx.plot(steps_k, y_mx, marker="o", color=colors[arch], label=arch_label[arch])

    # Red dotted minima per panel (no labels)
    ax_tr.axhline(best_train, linestyle=":", color="red", linewidth=1.5)
    ax_bx.axhline(best_biax,  linestyle=":", color="red", linewidth=1.5)
    ax_mx.axhline(best_mixed, linestyle=":", color="red", linewidth=1.5)

    # Formatting
    title_agg = "median" if agg == "median" else "mean"
    ax_tr.set_title(f"Task 2.2 — MS P-RMSE vs training steps ({title_agg} over inits)")

    for ax, ylabel in zip(
        (ax_tr, ax_bx, ax_mx),
        ("Training RMSE", "Test RMSE (biax)", "Test RMSE (mixed)"),
    ):
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    ax_mx.set_xlabel("Training steps [k]")
    ax_tr.legend(title="Architecture")

    fig.tight_layout()
    plt.show()

def plot_P_component_mirrored_density_grid(
    P_true: np.ndarray,
    P_pred_list,
    *,
    title: str = "",
    bins: int = 80,
    clip_percentiles=(0.5, 99.5),
    cmap: str = "viridis",
):
    """
    3x3 grid. Each cell is split into two vertical halves:
      left  = density of TRUE values for that P_ij component
      right = density of PRED values for that component

    The visualization in each cell is a heatmap over (value-bin, {true,pred}),
    i.e. a (bins x 2) image. Each column is normalized to sum to 1.

    Parameters
    ----------
    P_true : (N,3,3)
    P_pred_list :
        either list of K arrays each (N,3,3) or array (K,N,3,3) or (N,3,3).
        We aggregate over inits by stacking all predicted values.
    bins : number of y-bins
    clip_percentiles : robust range per component based on combined true+pred values
    """

    P_true = np.asarray(P_true)
    if P_true.ndim != 3 or P_true.shape[1:] != (3, 3):
        raise ValueError("P_true must have shape (N,3,3).")

    # Normalize P_pred input
    if isinstance(P_pred_list, list):
        P_pred = np.stack([np.asarray(p) for p in P_pred_list], axis=0)  # (K,N,3,3)
    else:
        P_pred = np.asarray(P_pred_list)
        if P_pred.ndim == 3:
            P_pred = P_pred[None, ...]  # (1,N,3,3)
    if P_pred.ndim != 4 or P_pred.shape[2:] != (3, 3):
        raise ValueError("P_pred_list must be list[(N,3,3)] or array (K,N,3,3) or (N,3,3).")

    # Precompute global vmax for consistent color scale within the figure
    # (since columns are normalized, values are in [0,1])
    H_all = []
    for i in range(3):
        for j in range(3):
            tvals = P_true[:, i, j].reshape(-1)
            pvals = P_pred[:, :, i, j].reshape(-1)  # aggregate across inits

            lo, hi = np.percentile(np.concatenate([tvals, pvals]), clip_percentiles)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                # fallback: expand slightly
                m = np.nanmedian(np.concatenate([tvals, pvals]))
                lo, hi = m - 1.0, m + 1.0

            # hist counts
            ct, edges = np.histogram(tvals, bins=bins, range=(lo, hi))
            cp, _     = np.histogram(pvals, bins=bins, range=(lo, hi))

            # normalize per column
            ct = ct.astype(float)
            cp = cp.astype(float)
            ct = ct / ct.sum() if ct.sum() > 0 else ct
            cp = cp / cp.sum() if cp.sum() > 0 else cp

            H = np.stack([ct, cp], axis=1)  # (bins,2)
            H_all.append(H)

    vmax = float(np.max(H_all)) if H_all else 1.0
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1.0

    # Plot grid
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.0), sharey=False)
    fig.suptitle(title, fontsize=14)

    im_for_cbar = None
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]

            tvals = P_true[:, i, j].reshape(-1)
            pvals = P_pred[:, :, i, j].reshape(-1)

            lo, hi = np.percentile(np.concatenate([tvals, pvals]), clip_percentiles)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                m = np.nanmedian(np.concatenate([tvals, pvals]))
                lo, hi = m - 1.0, m + 1.0

            ct, edges = np.histogram(tvals, bins=bins, range=(lo, hi))
            cp, _     = np.histogram(pvals, bins=bins, range=(lo, hi))

            ct = ct.astype(float); cp = cp.astype(float)
            ct = ct / ct.sum() if ct.sum() > 0 else ct
            cp = cp / cp.sum() if cp.sum() > 0 else cp
            H = np.stack([ct, cp], axis=1)  # (bins,2)

            # show as (y-bins x 2-columns) heatmap
            im = ax.imshow(
                H,
                origin="lower",
                aspect="auto",
                extent=(0, 2, edges[0], edges[-1]),
                vmin=0.0,
                vmax=vmax,
                cmap=cmap,
            )
            im_for_cbar = im

            # Subplot labels
            ax.set_title(f"P{i+1}{j+1}", fontsize=10)
            ax.set_xlim(0, 2)
            ax.set_xticks([0.5, 1.5])
            ax.set_xticklabels(["true", "pred"], fontsize=9)

            # vertical split line
            ax.axvline(1.0, color="white", linewidth=1.0, alpha=0.9)

            # reduce clutter
            if j != 0:
                ax.set_yticklabels([])
            else:
                ax.tick_params(axis="y", labelsize=9)

    # Single shared colorbar
    cbar = fig.colorbar(im_for_cbar, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    cbar.set_label("Normalized frequency (per column)", rotation=90)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def evaluate_task3_calibration_study(
    artifacts_dir: str | Path,
    dataset_1: dict,
    *,
    test_mode: str = "mixed",  # "biax", "mixed", "full"
    model_id: str = "WITI",
) -> dict[str, np.ndarray]:
    """
    Evaluate all Task 3 calibration study models and return component-wise RMSE
    for each training subset.
    
    Args:
        artifacts_dir: Path to task3_calibration_study artifacts folder
        dataset_1: Prepared dataset from workflows.prepare_dataset_1()
        test_mode: Which test set to use ("biax", "mixed", "full")
        model_id: Model type to evaluate (default "WITI")
        
    Returns:
        Dict mapping subset_tag -> (9,) array of component RMSE values
        e.g. {"biaxial": [0.01, 0.02, ...], "biaxial+uniaxial": [...], ...}
    
    Example:
        >>> component_errors = evaluate_task3_calibration_study(
        ...     "artifacts/task3_calibration_study",
        ...     dataset_1,
        ...     test_mode="mixed"
        ... )
        >>> # Then plot with visualization.plot_task3_component_heatmap(component_errors)
    """
    from pathlib import Path
    
    artifacts_dir = Path(artifacts_dir)
    
    # Load all runs
    runs = ewf.load_runs(
        artifacts_dir,
        model_id=model_id,
        dataset_1=dataset_1,
        strict=False,
    )
    
    if not runs:
        raise ValueError(f"No runs found in {artifacts_dir} for model_id={model_id}")
    
    # Group runs by subset_tag
    subset_groups = {}
    for r in runs:
        subset_tag = r.meta.get("subset_tag", "unknown")
        if subset_tag not in subset_groups:
            subset_groups[subset_tag] = []
        subset_groups[subset_tag].append(r)
    
    # Compute RMSE for each subset
    component_errors = {}
    
    for subset_tag, subset_runs in subset_groups.items():
        report = compute_rmse_over_test_set(
            subset_runs,
            dataset_1=dataset_1,
            test_mode=test_mode,
            model_name=subset_tag,
            return_component_metrics=True,
        )
        # rmse_P_comp_mean is (3,3) -> flatten to (9,)
        component_errors[subset_tag] = report.rmse_P_comp_mean.flatten()
    
    return component_errors

def collect_component_matrix_task2_2_ms(
    runs,
    *,
    dataset_1: dict,
    test_which: str = "full",
    metric: str = "bias",   # "bias" or "rmse"
    reduce: str = "median", # "median" (recommended) or "mean"
    arch_order=("small", "medium", "large"),
    steps_list=(100_000, 300_000, 500_000, 700_000, 900_000),
):
    """
    Returns:
      M[(arch, steps)] -> (3,3) matrix (bias or RMSE), reduced over inits
      vmax -> robust global scale (95th percentile), for consistent coloring
    """
    metric = metric.lower().strip()
    reduce = reduce.lower().strip()
    if metric not in ("bias", "rmse"):
        raise ValueError("metric must be 'bias' or 'rmse'.")
    if reduce not in ("median", "mean"):
        raise ValueError("reduce must be 'median' or 'mean'.")

    # test set once
    any_run = runs[0]
    X_test, Y_test = ewf.get_test_ms(any_run, dataset_1=dataset_1, which=test_which)
    P_true = Y_test.reshape(Y_test.shape[0], 3, 3)

    # group by (arch, steps)
    def _parse_arch_steps(r):
        arch = None
        for a in arch_order:
            if f"MS_{a}_" in r.tag:
                arch = a
                break
        steps = int(getattr(r, "steps", -1) or -1)
        if steps < 0:
            m = re.search(r"_steps(\d+)", r.tag)
            steps = int(m.group(1)) if m else -1
        if arch is None or steps < 0:
            return None
        return (arch, steps)

    groups = ewf.group_runs(runs, by=_parse_arch_steps)
    groups.pop(None, None)

    M = {}
    all_vals = []

    for arch in arch_order:
        for steps in steps_list:
            key = (arch, int(steps))
            cfg_runs = groups.get(key, [])
            if not cfg_runs:
                continue

            per_init = []
            for r in cfg_runs:
                P_pred = ewf.predict_ms_stress(r.model, X_test)  # (N,3,3)
                err = np.asarray(P_pred) - np.asarray(P_true)   # (N,3,3)

                if metric == "bias":
                    A = np.mean(err, axis=0)                    # (3,3)
                else:  # rmse
                    A = np.sqrt(np.mean(err**2, axis=0))        # (3,3)

                per_init.append(A)

            per_init = np.stack(per_init, axis=0)              # (K,3,3)
            A_red = np.median(per_init, axis=0) if reduce == "median" else np.mean(per_init, axis=0)

            M[key] = A_red
            all_vals.append(np.abs(A_red).reshape(-1) if metric == "bias" else A_red.reshape(-1))

    if not M:
        raise RuntimeError("No matrices computed. Check tag parsing and steps_list.")

    all_vals = np.concatenate(all_vals, axis=0)
    all_vals = all_vals[np.isfinite(all_vals)]
    vmax = float(np.percentile(all_vals, 95)) if all_vals.size else 1.0
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1.0

    return M, vmax


import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

def plot_component_tile_grid_arch_steps(
    M: dict,
    *,
    vmax: float,
    metric: str = "bias",          # "bias" or "rmse"
    arch_order=("small", "medium", "large"),
    steps_list=(100_000, 300_000, 500_000, 700_000, 900_000),
    title: str = "",
    cmap_bias="RdBu_r",
    cmap_rmse="viridis",
    figsize=(9.2, 11.0),
    show_component_labels: bool = False,
    log_color: bool = False,
    log_linthresh: float | None = None,   # only used for bias+log (SymLogNorm)
    wspace: float = 0.12,                 # NEW: reduce column gaps
):
    metric = metric.lower().strip()
    if metric not in ("bias", "rmse"):
        raise ValueError("metric must be 'bias' or 'rmse'.")

    # Column titles per your requested mapping
    arch_titles = {"small": "l=2, n=8", "medium": "l=3, n=16", "large": "l=4, n=32"}

    n_rows = len(steps_list)
    n_cols = len(arch_order)

    # Choose colormap + normalization
    if metric == "bias":
        cmap = cmap_bias
        if not log_color:
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        else:
            if log_linthresh is None:
                log_linthresh = max(vmax * 0.03, 1e-12)
            norm = mcolors.SymLogNorm(linthresh=log_linthresh, vmin=-vmax, vmax=vmax, base=10)
        cbar_label = r"bias = median$_{inits}$ mean$_{samples}$(pred − true)"
    else:
        cmap = cmap_rmse
        # RMSE is nonnegative; for log color use LogNorm, else plain Normalize
        if not log_color:
            norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
        else:
            vmin = max(vmax * 1e-4, 1e-12)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        cbar_label = r"RMSE = median$_{inits}$ sqrt(mean$_{samples}$((pred − true)$^2$))"

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=14)

    # Tighten inter-panel spacing (fixes the big empty column gaps)
    # tighter spacing between subplots (columns + rows)
    fig.subplots_adjust(
        left=0.08,
        right=0.86,   # keep room for the colorbar axis you add later
        top=0.93,
        bottom=0.06,
        wspace=wspace,  # <-- decrease to tighten columns (try 0.12–0.20)
        hspace=0.22,  # row spacing
    )


    im_for_cbar = None

    for r, steps in enumerate(steps_list):
        for c, arch in enumerate(arch_order):
            ax = axes[r, c]
            key = (arch, int(steps))
            A = M.get(key)

            if A is None:
                A = np.zeros((3, 3), dtype=float)
                ax.imshow(A, cmap=cmap, norm=norm)
                ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=14, transform=ax.transAxes)
            else:
                im_for_cbar = ax.imshow(A, cmap=cmap, norm=norm)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

            if r == 0:
                ax.set_title(arch_titles.get(arch, arch), fontsize=12, pad=8)
            if c == 0:
                ax.set_ylabel(f"{steps//1000}k", fontsize=11, rotation=0, labelpad=26, va="center")

            if show_component_labels:
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, f"$P_{{{i+1}{j+1}}}$",
                                ha="center", va="center", fontsize=8, color="black")

    # ---- Figure-level colorbar axis (does NOT affect any subplot geometry) ----
    # Reserve space on the right for the colorbar, then place it explicitly.
    # cbar_pad is the right margin reserved; increase it for more room.
    fig.subplots_adjust(right=0.88)  # <-- controls how much space remains for the grid

    # Add a dedicated axis for the colorbar: [left, bottom, width, height] in figure coords
    # Move it right by increasing 'left'; change thickness via 'width'.
    cax = fig.add_axes([0.90, 0.18, 0.02, 0.64])

    if im_for_cbar is None:
        im_for_cbar = axes[0, 0].imshow(np.zeros((3, 3)), cmap=cmap, norm=norm)

    cbar = fig.colorbar(im_for_cbar, cax=cax)
    cbar.set_label(cbar_label, rotation=90)


    plt.show()


import numpy as np
import re

def collect_component_matrix_task3_section2_witi(
    runs,
    *,
    dataset_1: dict,
    test_which: str = "full",       # "biax" | "mixed" | "full"
    metric: str = "bias",           # "bias" | "rmse"
    reduce: str = "median",         # "median" (recommended) | "mean"
    steps_list=(100_000, 300_000, 500_000, 700_000, 900_000),
    arch_order=("small", "medium", "large"),
):
    """
    Task 3 Section 2 (WITI) -> returns per (arch, steps) a (3,3) matrix for P-components.

    metric="bias":  median_inits(mean_samples(P_pred - P_true))   -> (3,3)
    metric="rmse":  median_inits(sqrt(mean_samples((P_pred - P_true)^2))) -> (3,3)

    Returns:
      M[(arch, steps)] -> (3,3)
      vmax -> robust global scale (95th percentile), for consistent coloring
    """
    metric = metric.lower().strip()
    reduce = reduce.lower().strip()
    if metric not in ("bias", "rmse"):
        raise ValueError("metric must be 'bias' or 'rmse'.")
    if reduce not in ("median", "mean"):
        raise ValueError("reduce must be 'median' or 'mean'.")

    # Build test set once
    any_run = runs[0]
    (F_test, I_test), (W_true, P_true) = ewf.get_test_witi(
        any_run, dataset_1=dataset_1, which=test_which
    )
    P_true = np.asarray(P_true)  # (N,3,3)

    # Map (l,n) -> architecture bucket to match your Task 2.2 column headers
    ln_to_arch = {(2, 8): "small", (3, 16): "medium", (4, 32): "large"}

    def _parse_arch_steps(r):
        m = re.search(r"_l(\d+)_n(\d+)_steps(\d+)", r.tag)
        if not m:
            return None
        l = int(m.group(1))
        n = int(m.group(2))
        steps = int(m.group(3))
        arch = ln_to_arch.get((l, n))
        if arch is None:
            return None
        return (arch, steps)

    groups = ewf.group_runs(runs, by=_parse_arch_steps)
    groups.pop(None, None)

    M = {}
    all_vals = []

    for arch in arch_order:
        for steps in steps_list:
            key = (arch, int(steps))
            cfg_runs = groups.get(key, [])
            if not cfg_runs:
                continue

            per_init = []
            for r in cfg_runs:
                # WITI model: ((F,I)) -> (Wp, Pp)
                Wp, Pp = jax.vmap(r.model)((F_test, I_test))  # Pp: (N,3,3)
                Pp = np.asarray(Pp)
                err = Pp - P_true  # (N,3,3)

                if metric == "bias":
                    A = np.mean(err, axis=0)                 # (3,3)
                else:
                    A = np.sqrt(np.mean(err**2, axis=0))     # (3,3)

                per_init.append(A)

            per_init = np.stack(per_init, axis=0)            # (K,3,3)
            A_red = np.median(per_init, axis=0) if reduce == "median" else np.mean(per_init, axis=0)

            M[key] = A_red
            all_vals.append(np.abs(A_red).reshape(-1) if metric == "bias" else A_red.reshape(-1))

    if not M:
        raise RuntimeError("No matrices computed. Check tag parsing, steps_list, and ln_to_arch mapping.")

    all_vals = np.concatenate(all_vals, axis=0)
    all_vals = all_vals[np.isfinite(all_vals)]
    vmax = float(np.percentile(all_vals, 95)) if all_vals.size else 1.0
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1.0

    return M, vmax

#Helper to obtain calibration data
#Once all trainings are implemented consistently we should add a standardzed utility that can do this for any model
def witi_calibration_provider_factory(ds1):
    def provider(label, res_obj):
        # Calibration set used in Task 3: [biaxial, uniaxial, pure_shear]
        F_bi  = ds1["F_bi"]
        F_uni = ds1["F_uni"]
        F_ps  = ds1["F_ps"]

        W_bi  = ds1["W_bi"]
        W_uni = ds1["W_uni"]
        W_ps  = ds1["W_ps"]

        F_cal = jnp.concatenate([jnp.array(F_bi), jnp.array(F_uni), jnp.array(F_ps)], axis=0)
        W_cal = jnp.concatenate([jnp.array(W_bi).reshape(-1), jnp.array(W_uni).reshape(-1), jnp.array(W_ps).reshape(-1)], axis=0)

        detF_cal = jnp.linalg.det(F_cal)
        return {"detF_cal": detF_cal, "W_cal": W_cal}
    return provider

# ---------------------------------------------------------------------
# Task 5.2 — Dataset 3 helpers (train/test are defined by prepare_dataset_3)
# ---------------------------------------------------------------------

def get_task5_2_train_test_sets(
    dataset_3: dict,
    *,
    G_cub: jnp.ndarray,            # kept for API compatibility; not used unless you later add fallbacks
    include_test_by_key: bool = False,
):
    """
    Returns the EXACT sets used by Task 5.2 training/eval.

    Train (interpolation): dataset_3["train_data_WI_cubic"]
      = ((F_cal, I_cal), ((W_cal, P_cal), weights_cal))

    Test (extrapolation): dataset_3["test_data_WI_cubic"]
      = ((F_test, I_test), (W_test, P_test))

    If include_test_by_key=True, also returns per-test-path splits WITHOUT recomputing invariants,
    by slicing the already-concatenated (F_test, I_test, W_test, P_test) according to the
    concatenation order used in prepare_dataset_3 (dataset_3["test_keys"]).
    """
    # --- train (calibration) ---
    (F_tr, I_tr), ((W_tr, P_tr), _weights_tr) = dataset_3["train_data_WI_cubic"]

    # --- test (full test concat) ---
    (F_te, I_te), (W_te, P_te) = dataset_3["test_data_WI_cubic"]

    out = {
        "train": ((F_tr, I_tr), (W_tr, P_tr)),
        "test":  ((F_te, I_te), (W_te, P_te)),
    }

    if include_test_by_key:
        # Slice per key using the same concatenation order as prepare_dataset_3:
        # F_test = concat([F_dict[k] for k in test_keys], axis=0)
        # and I_test computed once on that concatenation.
        test_by_key = {}
        cursor = 0
        for k in dataset_3["test_keys"]:
            n_k = int(dataset_3["F_dict"][k].shape[0])
            sl = slice(cursor, cursor + n_k)
            test_by_key[k] = ((F_te[sl], I_te[sl]), (W_te[sl], P_te[sl]))
            cursor += n_k

        out["test_by_key"] = test_by_key

    return out



def _rmse_scalar(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    y_true = jnp.ravel(y_true)
    y_pred = jnp.ravel(y_pred)
    return float(jnp.sqrt(jnp.mean((y_pred - y_true) ** 2)))


def _rmse_tensor(P_true: jnp.ndarray, P_pred: jnp.ndarray) -> float:
    # P_*: (N,3,3)
    return float(jnp.sqrt(jnp.mean((P_pred - P_true) ** 2)))


def _component_bias(P_true: jnp.ndarray, P_pred: jnp.ndarray) -> jnp.ndarray:
    # returns (3,3): mean(pred - true) over samples
    return jnp.mean(P_pred - P_true, axis=0)


def _component_rmse(P_true: jnp.ndarray, P_pred: jnp.ndarray) -> jnp.ndarray:
    # returns (3,3): rmse per component over samples
    return jnp.sqrt(jnp.mean((P_pred - P_true) ** 2, axis=0))


def _reduce_over_inits(mats: list[jnp.ndarray], reduce: str) -> jnp.ndarray:
    """
    mats: list of (3,3) arrays (or scalar arrays); reduce across list dimension.
    """
    X = jnp.stack(mats, axis=0)
    if reduce == "median":
        return jnp.median(X, axis=0)
    if reduce == "mean":
        return jnp.mean(X, axis=0)
    raise ValueError("reduce must be one of {'median','mean'}")


def collect_component_matrix_task5_2_wicub(
    runs_wicub: list,
    *,
    dataset_3: dict,
    G_cub: jnp.ndarray,   # kept for signature consistency; not used here
    metric: str = "bias",         # "bias" | "rmse"
    reduce: str = "median",       # "median" | "mean"
    steps_list=(100_000, 300_000),
    arch_order=("small", "medium"),
    batch_size: int = 256,        # performance
    max_samples: int | None = None,  # performance (optional)
):
    """
    Returns:
      M[(arch, steps)] -> (3,3) matrix
      vmax -> robust 95th percentile scale
    """
    metric = metric.lower().strip()
    reduce = reduce.lower().strip()

    sets = get_task5_2_train_test_sets(dataset_3, G_cub=G_cub, include_test_by_key=False)
    (F_te, I_te), (_W_te, P_te) = sets["test"]

    # optional subsample (helps a lot on WICUB)
    if max_samples is not None and F_te.shape[0] > max_samples:
        key = jrandom.PRNGKey(0)
        idx = jrandom.choice(key, F_te.shape[0], shape=(max_samples,), replace=False)
        F_te = F_te[idx]
        I_te = I_te[idx]
        P_te = P_te[idx]

    # Parse arch and steps robustly from your known tag format
    def _parse_arch_steps(tag: str):
        m = re.search(r"_((?:small|medium|large))_l\d+_n\d+_steps(\d+)", tag.lower())
        if not m:
            return None, None
        return m.group(1), int(m.group(2))

    buckets = {(a, int(s)): [] for a in arch_order for s in steps_list}
    for r in runs_wicub:
        arch, steps = _parse_arch_steps(r.tag)
        if arch in arch_order and steps in steps_list:
            buckets[(arch, steps)].append(r)

    M = {}
    all_vals = []

    for steps in steps_list:
        for arch in arch_order:
            rs = buckets[(arch, int(steps))]
            if not rs:
                M[(arch, int(steps))] = jnp.zeros((3, 3))
                continue

            per_init = []
            for rr in rs:
                # batched prediction (WICUB is expensive)
                _, Pp = predict_wicub_wp_batched(rr.model, F_te, I_te, batch_size=batch_size)

                if metric == "bias":
                    per_init.append(_component_bias(P_te, Pp))
                elif metric == "rmse":
                    per_init.append(_component_rmse(P_te, Pp))
                else:
                    raise ValueError("metric must be 'bias' or 'rmse'")

            M[(arch, int(steps))] = _reduce_over_inits(per_init, reduce=reduce)

            if metric == "bias":
                all_vals.append(jnp.abs(M[(arch, int(steps))]).reshape(-1))
            else:
                all_vals.append(M[(arch, int(steps))].reshape(-1))

    all_vals = jnp.concatenate(all_vals, axis=0) if all_vals else jnp.array([1.0])
    vmax = float(jnp.percentile(all_vals, 95))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    return M, vmax



# ---------------------------------------------------------------------
# Task 5.2 — 3-panel RMSE vs steps (train / test-all / test-per-path aggregate)
# ---------------------------------------------------------------------

def plot_task5_2_train_test_rmse_vs_steps(
    *,
    art_dir: str,
    dataset_3: dict,
    G_cub: jnp.ndarray,
    metric: str = "P",                 # "P" stress or "W" energy
    agg: str = "median",               # over inits
    steps_list=(100_000, 300_000),
    arch_order=("small", "medium"),
    title: str | None = None,
):
    """
    3 aligned panels:
      1) Train RMSE (calibration set)
      2) Test RMSE on concatenated test set (all test keys)
      3) Test RMSE aggregated per test-path (median over test_keys)

    Both y-axes are log.
    """

    runs = ewf.load_runs(art_dir, model_id="WICUB", dataset_3=dataset_3, strict=False, G_cub=G_cub)
    if not runs:
        raise FileNotFoundError(f"No WICUB runs found in {art_dir}")

    sets = get_task5_2_train_test_sets(dataset_3, G_cub=G_cub, include_test_by_key=True)
    (F_tr, I_tr), (W_tr, P_tr) = sets["train"]
    (F_te, I_te), (W_te, P_te) = sets["test"]
    test_by_key = sets["test_by_key"]

    def _parse_arch_steps(tag: str):
        m = re.search(r"_([a-zA-Z]+)_l\d+_n\d+_steps(\d+)", tag)
        if not m:
            return None, None
        return m.group(1).lower(), int(m.group(2))

    buckets = {(a, s): [] for a in arch_order for s in steps_list}
    for r in runs:
        arch, steps = _parse_arch_steps(r.tag)
        if arch in arch_order and steps in steps_list:
            buckets[(arch, steps)].append(r)

    def _rmse_for_one_model(rr, which: str):
        if which == "train":
            Wp, Pp = jax.vmap(rr.model)((F_tr, I_tr))
            return _rmse_tensor(P_tr, Pp) if metric == "P" else _rmse_scalar(W_tr, jnp.squeeze(Wp))
        if which == "test":
            Wp, Pp = jax.vmap(rr.model)((F_te, I_te))
            return _rmse_tensor(P_te, Pp) if metric == "P" else _rmse_scalar(W_te, jnp.squeeze(Wp))
        if which == "test_by_key_median":
            per_key = []
            for k, ((Fk, Ik), (Wk, Pk)) in test_by_key.items():
                Wp, Pp = jax.vmap(rr.model)((Fk, Ik))
                per_key.append(_rmse_tensor(Pk, Pp) if metric == "P" else _rmse_scalar(Wk, jnp.squeeze(Wp)))
            return float(np.median(per_key)) if per_key else float("nan")
        raise ValueError("unknown which")

    def _agg(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.nan
        return float(np.median(vals)) if agg == "median" else float(np.mean(vals))

    xs = [int(s // 1000) for s in steps_list]  # k steps for ticks

    series = {a: {"train": [], "test": [], "test_by_key_median": []} for a in arch_order}
    for a in arch_order:
        for s in steps_list:
            rs = buckets[(a, s)]
            train_vals = [_rmse_for_one_model(rr, "train") for rr in rs]
            test_vals  = [_rmse_for_one_model(rr, "test") for rr in rs]
            key_vals   = [_rmse_for_one_model(rr, "test_by_key_median") for rr in rs]

            series[a]["train"].append(_agg(train_vals))
            series[a]["test"].append(_agg(test_vals))
            series[a]["test_by_key_median"].append(_agg(key_vals))

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    if title is None:
        title = f"Task 5.2 — {metric}-RMSE vs training steps ({agg} over inits)"
    fig.suptitle(title)

    panels = [
        ("train", "Training RMSE (calibration)"),
        ("test", "Test RMSE (all test keys)"),
        ("test_by_key_median", "Test RMSE (median over test paths)"),
    ]

    for ax, (key, ylabel) in zip(axes, panels):
        for a in arch_order:
            ax.plot(xs, series[a][key], marker="o", label=a)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    axes[0].legend(title="Architecture", loc="upper right")
    axes[-1].set_xlabel("Training steps [k]")

    plt.tight_layout()
    plt.show()

def predict_wicub_energy_fast(model, I: jnp.ndarray) -> jnp.ndarray:
    """
    Fast W prediction that avoids model.__call__ (and thus avoids jacobian/grad).
    I: (N,6) -> returns (N,)
    """
    W = jax.vmap(model.nn)(I)
    return jnp.squeeze(W)

def predict_wicub_wp_batched(model, F: jnp.ndarray, I: jnp.ndarray, *, batch_size: int = 256):
    """
    Batched W,P prediction via model.__call__ (expensive; includes jacobians).
    Returns:
      W: (N,), P: (N,3,3)
    """
    N = F.shape[0]
    Ws = []
    Ps = []
    for i0 in range(0, N, batch_size):
        i1 = min(i0 + batch_size, N)
        Wb, Pb = jax.vmap(model)((F[i0:i1], I[i0:i1]))
        Ws.append(jnp.squeeze(Wb))
        Ps.append(Pb)
    return jnp.concatenate(Ws, axis=0), jnp.concatenate(Ps, axis=0)
