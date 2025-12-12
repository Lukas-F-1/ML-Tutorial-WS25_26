import jax
import jax.numpy as jnp
from . import data_t2 as td2
import matplotlib.pyplot as plt
import numpy as np

def evaluate_growth_condition(model, model_type: str, n: int = 100, eps_max: float = 1e-2):
    """
    Evaluate how well a model satisfies the growth condition near the identity.

    Parameters
    ----------
    model       : trained model object
    model_type  : "WI", "WI_Cubic", or "WF"
    n           : number of deformation samples (INCLUDING the identity)
    eps_max     : maximum perturbation epsilon for the uniaxial-like deformation

    Returns
    -------
    results : list of (F_i, W_i)
        F_i : (3,3) deformation gradient
        W_i : scalar predicted energy
    """
    #Defining Structural Tensors
    # Transversly isotropic
    G_ti = jnp.array([[4.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5]])

    # Cubic
    G_cub = td2.G_cub()

    # Linearly spaced epsilons from 0 (identity) to eps_max
    # jnp.linspace includes both endpoints, so eps=0 is guaranteed.
    epsilons = jnp.linspace(0.0, eps_max, n)

    results = []

    for eps in epsilons:
        # Uniaxial-like deformation: stretch in 11-direction
        F = jnp.array([
            [1.0 + eps, 0.0, 0.0],
            [0.0,       1.0, 0.0],
            [0.0,       0.0, 1.0]
        ])

        # -------------------------------------------------
        # Prepare model-specific inputs
        # -------------------------------------------------
        if model_type == "WI":
            # TI invariants (I1, J, -J, I4, I5)
            I = td2.compute_all_invariants(F=F, G_ti=G_ti)  # shape (5,)
            model_input = I

        elif model_type == "WI_Cubic":
            # Cubic invariants (I1, I2, J, -J, I7, I11)
            I = td2.compute_all_invariants_cubic(F=F,G_cub=G_cub)  # shape (6,)
            model_input = I

        elif model_type == "WF":
            # WF model takes F directly; cofF and detF are computed internally
            model_input = F

        else:
            raise ValueError(f"Unknown model_type '{model_type}'.")

        # -------------------------------------------------
        # Predict W(F)
        # -------------------------------------------------
        W_pred, P_pred = model(model_input)

        results.append((F, float(W_pred)))

    return results


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


def plot_W_vs_detF(
    model,
    model_type: str,
    F_test: jnp.ndarray,
    F_train: jnp.ndarray | None = None,
    G=None,
    title: str | None = None,
):
    """
    Analyze energy-based models by plotting W(F) vs det(F) for
    calibration and test data separately.

    Parameters
    ----------
    model : callable
        Trained model.

        - "WI"        : transversely isotropic invariant-based PANN
                        model((F, I)) -> (W, P)
        - "WI_Cubic"  : cubic invariant-based PANN
                        model((F, I)) -> (W, P)
        - "WF"        : deformation-gradient-based PANN
                        model(F) -> (W, P)

    model_type : {"WI", "WI_Cubic", "WF"}
        Type of model (determines how inputs are built from F).

    F_test : (N_test, 3, 3)
        Deformation gradients for test data.

    F_train : (N_train, 3, 3), optional
        Deformation gradients for calibration / training data.
        If None, only test data is plotted.

    G : structural tensor, optional
        Required for invariant-based cubic models ("WI_Cubic").
        Passed to your invariant computation routine.

    title : str, optional
        Plot title. If None, a default title is used.
    """

    assert model_type in ("WI", "WI_Cubic", "WF"), \
        "model_type must be 'WI', 'WI_Cubic', or 'WF'"

    # ---------- helper: evaluate W on a batch of F ----------

    def eval_W_batch(F_batch: jnp.ndarray) -> jnp.ndarray:
        """
        Compute W(F) for a batch of deformation gradients F_batch.
        Returns a 1D array of shape (N,).
        """
        if model_type == "WI":
            # TI invariants; assume a single-F function exists and vmap it.
            # You may adapt the function name to your code.
            I_batch = jax.vmap(td2.compute_all_invariants)(F_batch)  # (N, dimI)

            # forward call expects (F, I)
            W_batch, P_batch = jax.vmap(lambda F, I: model((F, I)))(
                F_batch, I_batch
            )

        elif model_type == "WI_Cubic":
            # Cubic invariants; your code already has this batch function.
            assert G is not None, "G must be provided for WI_Cubic"
            I_batch = td2.compute_all_invariants_cubic(F_batch, G)  # (N, dimI)

            W_batch, P_batch = jax.vmap(lambda F, I: model((F, I)))(
                F_batch, I_batch
            )

        elif model_type == "WF":
            # WF takes F directly and computes cofF, detF internally
            W_batch, P_batch = jax.vmap(model)(F_batch)

        else:
            raise ValueError(f"Unknown model_type '{model_type}'")

        # Make sure W is 1D
        W_batch = jnp.squeeze(W_batch)
        return W_batch

    # ---------- evaluate W and det(F) for train / test ----------

    # Test set
    W_test = eval_W_batch(F_test)
    detF_test = jnp.linalg.det(F_test)

    # Optional training set
    W_train = None
    detF_train = None
    if F_train is not None:
        W_train = eval_W_batch(F_train)
        detF_train = jnp.linalg.det(F_train)

    # ---------- plotting ----------

    plt.figure(figsize=(7, 5))

    if F_train is not None:
        # calibration data: green, semi-transparent circles
        plt.scatter(
            detF_train,
            W_train,
            s=20,
            c="tab:green",
            alpha=0.6,
            marker="o",
            label="Calibration data",
        )

    # test data: orange crosses
    plt.scatter(
        detF_test,
        W_test,
        s=25,
        c="tab:orange",
        alpha=0.9,
        marker="x",
        label="Test data",
    )

    plt.xlabel(r"$\det(F)$")
    plt.ylabel(r"$W(F)$ (Predicted Energy)")

    if title is None:
        title = f"W vs det(F) – {model_type} model"
    plt.title(title)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optionally return the underlying data if you want to reuse it
    return {
        "detF_train": detF_train,
        "W_train": W_train,
        "detF_test": detF_test,
        "W_test": W_test,
    }

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

#helper fct to get model predictions for eval
import jax
import jax.numpy as jnp

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

    Parameters
    ----------
    use_symlog : bool
        If True, use symmetric log scale (recommended for wide error ranges).
    symlog_linthresh : float
        Linear threshold around zero for symlog scale.
    """
    P_true = np.array(P_true)

    model_names = list(P_pred_dict.keys())
    num_models = len(model_names)

    # -------------------------------------------------
    # 1) Precompute all errors to get global min/max
    # -------------------------------------------------
    all_errors = []

    for name in model_names:
        preds = _to_stacked_preds(P_pred_dict[name])  # (K,N,3,3)

        if init_reduce == "mean":
            P_pred_red = preds.mean(axis=0)
        else:
            raise ValueError(f"Unknown init_reduce='{init_reduce}'")

        err = P_pred_red - P_true          # (N,3,3)
        all_errors.append(err.reshape(-1))

    all_errors = np.concatenate(all_errors)
    e_min, e_max = all_errors.min(), all_errors.max()

    # Add small padding so boxes don’t touch borders
    pad = 0.05 * max(abs(e_min), abs(e_max))
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
                preds = _to_stacked_preds(P_pred_dict[name])
                P_pred_red = preds.mean(axis=0)
                e = (P_pred_red - P_true)[:, i, j]
                data.append(e)

            bp = ax.boxplot(data, patch_artist=True, showfliers=True)
            for patch in bp["boxes"]:
                patch.set_alpha(0.6)

            ax.set_ylim(y_min, y_max)

            if use_symlog:
                ax.set_yscale("symlog", linthresh=symlog_linthresh)

            ax.set_xticks(range(1, num_models + 1))
            ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
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

    Supports multiple initializations per model:
      - P_pred_dict[name] can be (N,3,3) OR (K,N,3,3) OR list of (N,3,3).
    Reduces init dimension by averaging predictions across inits first
    (so RMSE is computed on the averaged prediction).
    """
    P_true = np.array(P_true)
    model_names = list(P_pred_dict.keys())
    num_models = len(model_names)

    rmse = {}  # name -> (9,)
    for name in model_names:
        preds = _to_stacked_preds(P_pred_dict[name])  # (K,N,3,3)

        if init_reduce == "mean":
            P_pred_red = preds.mean(axis=0)          # (N,3,3)
        else:
            raise ValueError(f"Unknown init_reduce='{init_reduce}'")

        e = P_pred_red - P_true                      # (N,3,3)
        rmse_mat = np.sqrt(np.mean(e**2, axis=0))    # (3,3)
        rmse[name] = rmse_mat.reshape(-1)            # (9,)

    comp_labels_flat = ["11", "12", "13",
                        "21", "22", "23",
                        "31", "32", "33"]

    x = np.arange(9)
    width = 0.8 / num_models

    plt.figure(figsize=(12, 5))
    for idx, name in enumerate(model_names):
        offset = (idx - (num_models - 1) / 2) * width
        plt.bar(x + offset, rmse[name], width=width, label=name, alpha=0.8)

    plt.xticks(x, [f"P{c}" for c in comp_labels_flat])
    plt.ylabel("RMSE")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
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

    Parameters
    ----------
    W_true : (N,) or (N,1)
        Ground-truth energy.
    P_true : (N,3,3)
        Ground-truth stress.
    W_pred_dict : dict[str, preds]
        preds can be:
          - (N,) or (N,1)
          - (K,N) or (K,N,1)
          - list of (N,) arrays
    P_pred_dict : dict[str, preds]
        preds can be:
          - (N,3,3)
          - (K,N,3,3)
          - list of (N,3,3) arrays
    """

    W_true = np.squeeze(np.array(W_true))        # (N,)
    P_true = np.array(P_true)                    # (N,3,3)

    model_names = list(W_pred_dict.keys())

    rmse_W = {}
    rmse_P = {}

    for name in model_names:
        # -----------------
        # Energy RMSE
        # -----------------
        W_preds = _stack_preds(W_pred_dict[name])   # (K,N) or (K,N,1)
        W_preds = np.squeeze(W_preds)                # (K,N)
        W_mean = W_preds.mean(axis=0)                # (N,)

        rmse_W[name] = np.sqrt(np.mean((W_mean - W_true) ** 2))

        # -----------------
        # Stress RMSE
        # -----------------
        P_preds = _stack_preds(P_pred_dict[name])    # (K,N,3,3)
        P_mean = P_preds.mean(axis=0)                # (N,3,3)

        err = P_mean - P_true
        rmse_P[name] = np.sqrt(np.mean(err ** 2))    # scalar over all comps

    # -----------------
    # Plot: Energy RMSE
    # -----------------
    plt.figure(figsize=(6, 4))
    plt.bar(rmse_W.keys(), rmse_W.values(), alpha=0.8)
    plt.ylabel("RMSE (Energy W)")
    plt.title(title_energy)
    plt.yscale("log") 
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # -----------------
    # Plot: Stress RMSE
    # -----------------
    plt.figure(figsize=(6, 4))
    plt.bar(rmse_P.keys(), rmse_P.values(), alpha=0.8)
    plt.ylabel("RMSE (Stress P)")
    plt.yscale("log") 
    plt.title(title_stress)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
