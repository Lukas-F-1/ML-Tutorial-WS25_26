import jax
import jax.numpy as jnp
from . import data_t2 as td2
import matplotlib.pyplot as plt

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
            I = td2.compute_invariants_ti(F)  # shape (5,)
            model_input = I

        elif model_type == "WI_Cubic":
            # Cubic invariants (I1, I2, J, -J, I7, I11)
            I = td2.compute_all_invariants_cubic_single(F)  # shape (6,)
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
        I = td2.compute_invariants_ti(F)  # shape (5,)
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
            I_batch = jax.vmap(td2.compute_invariants_ti)(F_batch)  # (N, dimI)

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
