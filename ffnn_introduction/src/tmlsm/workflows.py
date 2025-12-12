from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import json
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import klax

from . import data_t2 as td2
from . import models as tm
from . import losses as tl
import pickle
from pathlib import Path


#Helper Functions

def _save_history(history, filepath: str | Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(history, f)


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_eqx_model(model, filepath: str | Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(filepath), model)


def prepare_dataset_1(
    *,
    base_path: str | Path | None = None,
    G_ti: jnp.ndarray | None = None,
    master_key: jrandom.PRNGKey = jrandom.PRNGKey(0),
):
    """
    Prepare Dataset 1 exactly like in your notebook, but packaged for reuse.

    Returns a dict with:
      - structural tensors: G_ti, G_cub
      - key: master_key
      - raw and jax arrays: F_*, P_*, W_*, C_*
      - invariants: I_* (TI invariants with [-J] included, dim=5)
      - reference invariants (biaxial) for validation
      - base_path (string)

    Notes:
      - Adds C_mix_test and its invariants too (useful later).
    """
    # Structural tensors
    if G_ti is None:
        G_ti = jnp.array([[4.0, 0.0, 0.0],
                          [0.0, 0.5, 0.0],
                          [0.0, 0.0, 0.5]])
    G_cub = td2.G_cub()

    # Resolve base_path like your notebook
    if base_path is None:
        current_folder = Path(os.getcwd())
        base_path = current_folder.parent / "hyperelasticity" / "data"
    base_path = str(base_path)

    # Load calibration + test
    F_uni, P_uni, W_uni = td2.load_hyperelastic_data(os.path.join(base_path, r"calibration\uniaxial.txt"))
    F_ps,  P_ps,  W_ps  = td2.load_hyperelastic_data(os.path.join(base_path, r"calibration\pure_shear.txt"))
    F_bi,  P_bi,  W_bi  = td2.load_hyperelastic_data(os.path.join(base_path, r"calibration\biaxial.txt"))

    F_bi_test,  P_bi_test,  W_bi_test  = td2.load_hyperelastic_data(os.path.join(base_path, r"test\biax_test.txt"))
    F_mix_test, P_mix_test, W_mix_test = td2.load_hyperelastic_data(os.path.join(base_path, r"test\mixed_test.txt"))

    # Right Cauchy-Green tensors (numpy)
    C_uni = F_uni.transpose(0, 2, 1) @ F_uni
    C_ps  = F_ps.transpose(0, 2, 1) @ F_ps
    C_bi  = F_bi.transpose(0, 2, 1) @ F_bi
    C_bi_test  = F_bi_test.transpose(0, 2, 1) @ F_bi_test
    C_mix_test = F_mix_test.transpose(0, 2, 1) @ F_mix_test

    # Convert to JAX arrays
    data = {
        "base_path": base_path,
        "G_ti": G_ti,
        "G_cub": G_cub,
        "master_key": master_key,

        "F_uni": jnp.array(F_uni), "P_uni": jnp.array(P_uni), "W_uni": jnp.array(W_uni), "C_uni": jnp.array(C_uni),
        "F_ps":  jnp.array(F_ps),  "P_ps":  jnp.array(P_ps),  "W_ps":  jnp.array(W_ps),  "C_ps":  jnp.array(C_ps),
        "F_bi":  jnp.array(F_bi),  "P_bi":  jnp.array(P_bi),  "W_bi":  jnp.array(W_bi),  "C_bi":  jnp.array(C_bi),

        "F_bi_test":  jnp.array(F_bi_test),  "P_bi_test":  jnp.array(P_bi_test),  "W_bi_test":  jnp.array(W_bi_test),  "C_bi_test":  jnp.array(C_bi_test),
        "F_mix_test": jnp.array(F_mix_test), "P_mix_test": jnp.array(P_mix_test), "W_mix_test": jnp.array(W_mix_test), "C_mix_test": jnp.array(C_mix_test),
    }

    # Compute TI invariants (dim=5: [I1, J, -J, I4, I5])
    data["I_uni"]     = td2.compute_all_invariants(data["F_uni"], G_ti)
    data["I_ps"]      = td2.compute_all_invariants(data["F_ps"],  G_ti)
    data["I_bi"]      = td2.compute_all_invariants(data["F_bi"],  G_ti)
    data["I_bi_test"] = td2.compute_all_invariants(data["F_bi_test"], G_ti)
    data["I_mix_test"]= td2.compute_all_invariants(data["F_mix_test"], G_ti)

    # Reference invariants for checking
    data["invariants_reference_biaxial"] = td2.load_invariants(
        os.path.join(base_path, r"invariants\I_biaxial.txt")
    )

    return data


#Workflows

def workflow_task_2_2_train_ms_sweep(
    dataset_1: dict,
    *,
    out_dir: str | Path = "artifacts/task2_2",
    batch_size: int = 32,
    learning_rate: float = 1e-3,
):
    """
    Task 2.2: Train MS(C)->P with multiple architectures and training steps.

    Saves:
      - model:  <out_dir>/MS_<arch>_l{l}_n{n}_steps{steps}.eqx
      - meta:   <out_dir>/MS_<arch>_l{l}_n{n}_steps{steps}.json

    Returns a list of dicts (one per trained model) with paths + metrics.
    """

    out_dir = _ensure_dir(out_dir)

    # ----- Build calibration dataset (same as your code) -----
    C_cal_MS = jnp.concatenate([dataset_1["C_uni"], dataset_1["C_ps"], dataset_1["C_bi"]], axis=0)
    P_cal_MS = jnp.concatenate([dataset_1["P_uni"], dataset_1["P_ps"], dataset_1["P_bi"]], axis=0)

    X_cal_MS = jax.vmap(td2.C_to_six)(C_cal_MS)                 # (N,6)
    Y_cal_MS = P_cal_MS.reshape(P_cal_MS.shape[0], 9)           # (N,9)
    train_data_MS = (X_cal_MS, Y_cal_MS)

    # ----- Test sets (same transformation, evaluate consistently) -----
    X_bi_test  = jax.vmap(td2.C_to_six)(dataset_1["C_bi_test"])
    Y_bi_test  = dataset_1["P_bi_test"].reshape(dataset_1["P_bi_test"].shape[0], 9)

    X_mix_test = jax.vmap(td2.C_to_six)(dataset_1["C_mix_test"])
    Y_mix_test = dataset_1["P_mix_test"].reshape(dataset_1["P_mix_test"].shape[0], 9)

    # Architectures to sweep
    archs = [
        ("small",  2,  8),
        ("medium", 3, 16),
        ("large",  4, 32),
    ]
    steps_list = [100_000, 300_000, 500_000]

    results = []

    master_key = dataset_1["master_key"]
    base_seed = int(jrandom.randint(master_key, (), 0, 10_000_000))

    run_idx = 0
    for arch_name, l, n in archs:
        for steps in steps_list:
            run_idx += 1

            # deterministic-ish keys per config
            cfg_seed = base_seed + 1000 * run_idx + 10 * l + n + steps // 1000
            model_key = jrandom.PRNGKey(cfg_seed + 1)
            train_key = jrandom.PRNGKey(cfg_seed + 2)

            # Build model
            MS_model = tm.build(
                key=model_key,
                input_dim=6,
                output_dim=9,
                num_hidden_layers=l,
                nodes_per_layer=n,
                activations=jax.nn.softplus,
                constrain_icnn_weights=False
            )

            # Train
            MS_trained, MS_history = tm.train_model(
                MS_model,
                train_data_MS,
                train_key,
                steps=steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                loss_fn=tl.MSE(),   # Task 2.2 baseline; swap to tl.WeightedMSE() if desired
            )

            MS_final = klax.finalize(MS_trained)

            # Evaluate RMSE on both test sets (vectorized)
            P_bi_pred  = jax.vmap(MS_final)(X_bi_test)
            P_mix_pred = jax.vmap(MS_final)(X_mix_test)

            rmse_bi  = float(jnp.sqrt(jnp.mean((P_bi_pred  - Y_bi_test)  ** 2)))
            rmse_mix = float(jnp.sqrt(jnp.mean((P_mix_pred - Y_mix_test) ** 2)))

                        # Save model + metadata + history
            tag = f"MS_{arch_name}_l{l}_n{n}_steps{steps}"
            model_path   = out_dir / f"{tag}.eqx"
            meta_path    = out_dir / f"{tag}.json"
            history_path = out_dir / f"{tag}_history.pkl"

            _save_eqx_model(MS_final, model_path)
            _save_history(MS_history, history_path)

            meta = {
                "model_id": "MS",
                "tag": tag,
                "arch_name": arch_name,
                "num_hidden_layers": l,
                "nodes_per_layer": n,
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "activation": "softplus",
                "train_dataset": "Dataset 1 calibration (uni + pure_shear + biax)",
                "test_sets": ["biax_test", "mixed_test"],
                "rmse_biax_test": rmse_bi,
                "rmse_mixed_test": rmse_mix,
                "saved_model_path": str(model_path),
                "saved_history_path": str(history_path),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            results.append(meta)


    return results

def workflow_task_2_3_train_ms_weighted_5inits(
    dataset_1: dict,
    *,
    out_dir: str | Path = "artifacts/task2_3",
    n_inits: int = 5,
    steps: int = 300_000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
):
    """
    Task 2.3: Loss-weighted MS model (MS_w) training with multiple random initializations.

    Model:
      MS_w(C)->P, trained with loss-weighted strategy (inverse path weights)
      Architecture = MEDIUM baseline from Task 2.2:
        - num_hidden_layers (l) = 3
        - nodes_per_layer  (n) = 16
        - activation = softplus
      Steps = 300000

    Compare against:
      Task 2.2 unweighted MS model with the SAME architecture (l=3,n=16) and steps=300000.

    Saves per run:
      - model:   MSW_medium_l3_n16_steps300000_init{k}.eqx
      - history: MSW_medium_l3_n16_steps300000_init{k}_history.pkl
      - meta:    MSW_medium_l3_n16_steps300000_init{k}.json
    """
    out_dir = _ensure_dir(out_dir)

    # ------------------------------------------------------------
    # Build calibration dataset (same as Task 2.2)
    # ------------------------------------------------------------
    C_cal_MS = jnp.concatenate([dataset_1["C_uni"], dataset_1["C_ps"], dataset_1["C_bi"]], axis=0)
    P_cal_MS = jnp.concatenate([dataset_1["P_uni"], dataset_1["P_ps"], dataset_1["P_bi"]], axis=0)

    X_cal_MS = jax.vmap(td2.C_to_six)(C_cal_MS)               # (N,6)
    Y_cal_MS = P_cal_MS.reshape(P_cal_MS.shape[0], 9)         # (N,9)

    # ------------------------------------------------------------
    # Compute inverse path weights (Task 2.3 logic)
    # IMPORTANT: order must match the concatenation used above:
    #   [uni, ps, bi] in both C_cal_MS and P_cal_MS
    # ------------------------------------------------------------
    P_uni = dataset_1["P_uni"]
    P_ps  = dataset_1["P_ps"]
    P_bi  = dataset_1["P_bi"]

    w_uni = td2.compute_path_weight(P_uni)
    w_ps  = td2.compute_path_weight(P_ps)
    w_bi  = td2.compute_path_weight(P_bi)

    w_uni_inv = 1.0 / w_uni
    w_ps_inv  = 1.0 / w_ps
    w_bi_inv  = 1.0 / w_bi

    weights_uni = w_uni_inv * jnp.ones(P_uni.shape[0])
    weights_ps  = w_ps_inv  * jnp.ones(P_ps.shape[0])
    weights_bi  = w_bi_inv  * jnp.ones(P_bi.shape[0])

    sample_weights = jnp.concatenate([weights_uni, weights_ps, weights_bi], axis=0).reshape(-1)

    # Weighted training data format: (X, (Y, weights))
    train_data_MS_w = (X_cal_MS, (Y_cal_MS, sample_weights))

    # Loss function: weighted MSE
    loss_fn_MS_w = tl.WeightedMSE()

    # ------------------------------------------------------------
    # Train n_inits independent initializations
    # ------------------------------------------------------------
    results = []
    master_key = dataset_1["master_key"]

    # Create deterministic subkeys for reproducibility
    keys = jrandom.split(master_key, n_inits * 2 + 1)
    base_key = keys[0]

    for k in range(n_inits):
        model_key = keys[1 + 2 * k]
        train_key = keys[1 + 2 * k + 1]

        # --------------------------------------------------------
        # MEDIUM architecture from Task 2.2:
        # l=3 hidden layers, n=16 nodes per layer, softplus
        # --------------------------------------------------------
        MS_w_model = tm.build(
            key=model_key,
            input_dim=6,
            output_dim=9,
            num_hidden_layers=3,
            nodes_per_layer=16,
            activations=jax.nn.softplus,
            constrain_icnn_weights=False
        )

        trained_MS_w, history_MS_w = tm.train_model(
            MS_w_model,
            train_data_MS_w,
            train_key,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_fn=loss_fn_MS_w
        )

        MS_w_final = klax.finalize(trained_MS_w)

        # Save artifacts
        tag = f"MSW_medium_l3_n16_steps{steps}_init{k+1:02d}"
        model_path   = out_dir / f"{tag}.eqx"
        history_path = out_dir / f"{tag}_history.pkl"
        meta_path    = out_dir / f"{tag}.json"

        _save_eqx_model(MS_w_final, model_path)
        _save_history(history_MS_w, history_path)

        meta = {
            "task": "2.3",
            "model_id": "MSW",
            "tag": tag,
            "architecture_note": "MEDIUM baseline from Task 2.2: l=3, n=16, softplus",
            "comparison_note": "Compare to Task 2.2 MS (unweighted) with l=3, n=16 and steps=300000",
            "num_hidden_layers": 3,
            "nodes_per_layer": 16,
            "activation": "softplus",
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss": "WeightedMSE",
            "weights": {
                "w_uni": float(w_uni),
                "w_ps": float(w_ps),
                "w_bi": float(w_bi),
                "w_uni_inv": float(w_uni_inv),
                "w_ps_inv": float(w_ps_inv),
                "w_bi_inv": float(w_bi_inv),
            },
            "saved_model_path": str(model_path),
            "saved_history_path": str(history_path),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        results.append(meta)

    return results

def workflow_task_3_train_wi_ti_strategies_abc(
    dataset_1: dict,
    *,
    out_dir: str | Path = "artifacts/task3",
    n_inits: int = 5,
    steps: int = 300_000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
):
    """
    Task 3 workflow: Train WI_ti benchmark model under strategies A/B/C with 5 random initializations each.

    Benchmark architecture (for clean comparison):
      - num_hidden_layers (l) = 3
      - nodes_per_layer  (n) = 16
      - activation       = softplus
      - steps            = 300000

    Strategies (WeightedSobolevLoss):
      - A: alpha=1, beta=0  (energy only)
      - B: alpha=0, beta=1  (stress/gradient only)
      - C: alpha=1, beta=1  (combined)

    Loss-weighting:
      Uses inverse path weights (1 / w_path) computed from calibration stresses
      (uniaxial, pure shear, biaxial).

    Saves per run:
      - model:   WITI_<A|B|C>_bench_l3_n16_steps{steps}_initXX.eqx
      - history: WITI_<A|B|C>_bench_l3_n16_steps{steps}_initXX_history.pkl
      - meta:    WITI_<A|B|C>_bench_l3_n16_steps{steps}_initXX.json
    """
    out_dir = _ensure_dir(out_dir)

    G_ti = dataset_1["G_ti"]
    master_key = dataset_1["master_key"]

    # ------------------------------------------------------------
    # Build calibration dataset with the order you used in Task 3:
    # IMPORTANT: order [biaxial, uniaxial, pure_shear]
    # ------------------------------------------------------------
    F_bi = dataset_1["F_bi"]
    F_uni = dataset_1["F_uni"]
    F_ps = dataset_1["F_ps"]

    P_bi = dataset_1["P_bi"]
    P_uni = dataset_1["P_uni"]
    P_ps = dataset_1["P_ps"]

    W_bi = dataset_1["W_bi"]
    W_uni = dataset_1["W_uni"]
    W_ps = dataset_1["W_ps"]

    # invariants (TI invariants) also need to be concatenated in the same order
    I_bi = dataset_1["I_bi"]
    I_uni = dataset_1["I_uni"]
    I_ps = dataset_1["I_ps"]

    F_cal_all = jnp.concatenate([F_bi, F_uni, F_ps], axis=0)
    I_cal_all = jnp.concatenate([I_bi, I_uni, I_ps], axis=0)
    W_cal_all = jnp.concatenate([W_bi, W_uni, W_ps], axis=0)
    P_cal_all = jnp.concatenate([P_bi, P_uni, P_ps], axis=0)

    # ------------------------------------------------------------
    # Compute inverse path weights from calibration paths (Task 3)
    # Path weights computed from stress magnitude along each path
    # ------------------------------------------------------------
    w_uni = td2.compute_path_weight(P_uni)
    w_ps  = td2.compute_path_weight(P_ps)
    w_bi  = td2.compute_path_weight(P_bi)

    w_uni_inv = 1.0 / w_uni
    w_ps_inv  = 1.0 / w_ps
    w_bi_inv  = 1.0 / w_bi

    weights_bi  = w_bi_inv  * jnp.ones(P_bi.shape[0])
    weights_uni = w_uni_inv * jnp.ones(P_uni.shape[0])
    weights_ps  = w_ps_inv  * jnp.ones(P_ps.shape[0])

    # IMPORTANT: must match F_cal_all concatenation order [bi, uni, ps]
    sample_weights = jnp.concatenate([weights_bi, weights_uni, weights_ps], axis=0).reshape(-1)

    # Training data format for WeightedSobolevLoss:
    # batch = (x, ((W_true, P_true), w))
    train_data = (
        (F_cal_all, I_cal_all),
        ((W_cal_all, P_cal_all), sample_weights)
    )

    # ------------------------------------------------------------
    # Define loss strategies
    # ------------------------------------------------------------
    strategies = {
        "A": tl.WeightedSobolevLoss(alpha=1.0, beta=0.0),
        "B": tl.WeightedSobolevLoss(alpha=0.0, beta=1.0),
        "C": tl.WeightedSobolevLoss(alpha=1.0, beta=1.0),
    }

    # ------------------------------------------------------------
    # Train 5 inits per strategy
    # ------------------------------------------------------------
    results = []
    # produce stable subkeys
    keys = jrandom.split(master_key, n_inits * 2 * len(strategies) + 1)
    key_cursor = 1

    for strat_name, loss_fn in strategies.items():
        for init_idx in range(n_inits):
            model_key = keys[key_cursor]; train_key = keys[key_cursor + 1]
            key_cursor += 2

            # --------------------------------------------------------
            # Benchmark WI_ti model:
            #   input_dim=5 invariants, output scalar W (grad used internally)
            #   l=3, n=16 (benchmark), softplus, FICNN enabled
            # --------------------------------------------------------
            WI_ti_model = tm.SobolevModel_WI_ti(
                G_ti=G_ti,
                key=model_key,
                input_dim=5,
                output_dim="scalar",
                num_hidden_layers=3,
                nodes_per_layer=16,
                activation=jax.nn.softplus,
                is_icnn=False,
                is_ficnn=True
            )

            trained_model, history = tm.train_WI(
                model=WI_ti_model,
                train_data=train_data,
                key=train_key,
                steps=steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                loss_fn=loss_fn
            )

            final_model = klax.finalize(trained_model)

            # Save artifacts
            tag = f"WITI_{strat_name}_bench_l3_n16_steps{steps}_init{init_idx+1:02d}"
            model_path   = out_dir / f"{tag}.eqx"
            history_path = out_dir / f"{tag}_history.pkl"
            meta_path    = out_dir / f"{tag}.json"

            _save_eqx_model(final_model, model_path)
            _save_history(history, history_path)

            meta = {
                "task": "3",
                "model_id": "WITI",
                "strategy": strat_name,
                "tag": tag,
                "benchmark_architecture": {"l": 3, "n": 16, "activation": "softplus"},
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "loss": "WeightedSobolevLoss",
                "loss_params": {
                    "alpha": float(loss_fn.alpha),
                    "beta": float(loss_fn.beta),
                },
                "loss_weighting": "inverse path weights (uniaxial, pure shear, biaxial)",
                "path_weights": {
                    "w_uni": float(w_uni), "w_ps": float(w_ps), "w_bi": float(w_bi),
                    "w_uni_inv": float(w_uni_inv), "w_ps_inv": float(w_ps_inv), "w_bi_inv": float(w_bi_inv),
                    "concat_order": "[biaxial, uniaxial, pure_shear]",
                },
                "saved_model_path": str(model_path),
                "saved_history_path": str(history_path),
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            results.append(meta)

    return results
