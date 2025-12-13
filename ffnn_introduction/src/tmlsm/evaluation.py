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
    rmse_P_std: float

    rmse_P_comp_per_init: np.ndarray     # (K, 3, 3)
    rmse_P_comp_mean: np.ndarray         # (3, 3)
    rmse_P_comp_std: np.ndarray          # (3, 3)

    bias_P_comp_per_init: np.ndarray     # (K, 3, 3)  mean signed error per init
    bias_P_comp_mean: np.ndarray         # (3, 3)
    bias_P_comp_std: np.ndarray          # (3, 3)

    # Energy metrics (optional; None if not available)
    rmse_W_per_init: np.ndarray | None   # (K,) or None
    rmse_W_mean: float | None
    rmse_W_std: float | None

    # Optional raw error tensors (can be large)
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
    """
    Compute RMSE (and signed error/bias) over the chosen test set for a group of runs
    (typically multiple random initializations of the same base model).

    Parameters
    ----------
    runs:
        Iterable of Run-like objects with at least:
          - .model_id (str)
          - .model (callable)
          - .meta_path (Path)
          - .tag (str)
        Typically: list[eval_workflows.Run]

    dataset_1, G_cub:
        Passed through to ewf.get_test_sets, which delegates to wf.get_test_data_for_run. :contentReference[oaicite:3]{index=3}

    test_mode:
        "biax", "mixed", or "full" (concatenate biax+mixed).

    return_component_metrics:
        If True, compute per-component RMSE (3x3) and bias (mean signed error).

    return_raw_errors:
        If True, return the raw error tensors:
           errors_P: (K,N,3,3) and errors_W: (K,N)
        This can be large; keep False unless you need histograms/parity later.

    Returns
    -------
    RMSEReport
    """
    runs = list(runs)
    if not runs:
        raise ValueError("compute_rmse_over_test_set received an empty runs iterable.")

    # Determine model_id/name from first run
    r0 = runs[0]
    mid = str(getattr(r0, "model_id", "")).upper()
    if model_name is None:
        model_name = getattr(r0, "base_tag", None) or getattr(r0, "tag", "model")

    # --- Build test set once, using run0 meta (consistent with your workflow style) ---
    ts = ewf.get_test_sets(r0, dataset_1=dataset_1, G_cub=G_cub)  # :contentReference[oaicite:4]{index=4}
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

        # Stress errors
        P_err = P_pred_np - P_true_np   # (N,3,3)
        P_errs.append(P_err)

        # Energy errors (if available on both sides)
        if W_true_np is not None and W_pred is not None:
            W_pred_np = np.squeeze(_as_np(W_pred))
            W_errs.append(W_pred_np - np.squeeze(W_true_np))

    P_errs = np.stack(P_errs, axis=0)  # (K,N,3,3)
    K = P_errs.shape[0]

    # --- Global stress RMSE per init (scalar over all entries) ---
    rmse_P_per_init = np.sqrt(np.mean(P_errs**2, axis=(1, 2, 3)))  # (K,)
    rmse_P_mean = float(np.mean(rmse_P_per_init))
    rmse_P_std  = float(np.std(rmse_P_per_init))

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
        rmse_W_mean = float(np.mean(rmse_W_per_init))
        rmse_W_std  = float(np.std(rmse_W_per_init))
        errors_W_out = W_errs if return_raw_errors else None
    else:
        rmse_W_per_init = None
        rmse_W_mean = None
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
        rmse_P_std=rmse_P_std,
        rmse_P_comp_per_init=rmse_P_comp_per_init,
        rmse_P_comp_mean=rmse_P_comp_mean,
        rmse_P_comp_std=rmse_P_comp_std,
        bias_P_comp_per_init=bias_P_comp_per_init,
        bias_P_comp_mean=bias_P_comp_mean,
        bias_P_comp_std=bias_P_comp_std,
        rmse_W_per_init=rmse_W_per_init,
        rmse_W_mean=rmse_W_mean,
        rmse_W_std=rmse_W_std,
        errors_P=errors_P_out,
        errors_W=errors_W_out,
    )

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
