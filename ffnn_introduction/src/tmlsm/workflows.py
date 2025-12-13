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
import itertools

from . import data_t2 as td2
from . import models as tm
from . import losses as tl
import pickle
from typing import Any, Iterable, Literal


#-----------------------Helper Functions

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

def prepare_dataset_4(
    *,
    base_path: str | Path | None = None,
    G_ti: jnp.ndarray | None = None,
):
    """Prepare Dataset 4 (concentric sampled deformation gradients)."""
    if G_ti is None:
        G_ti = jnp.array([[4.0, 0.0, 0.0],
                          [0.0, 0.5, 0.0],
                          [0.0, 0.0, 0.5]])
    
    if base_path is None:
        current_folder = Path(os.getcwd())
        base_path = current_folder.parent / "hyperelasticity" / "data"
    base_path = str(base_path)
    
    all_F = td2.load_all_concentric(base_path)
    all_C, all_I, all_W, all_P, inv_weights = td2.preprocess_all_concentric(all_F, G_ti)
    
    return {
        "base_path": base_path,
        "G_ti": G_ti,
        "all_F": all_F,
        "all_C": all_C,
        "all_I": all_I,
        "all_W": all_W,
        "all_P": all_P,
        "inv_weights": inv_weights,
    }

def _run_single_task4(args):
    """Single training for Task 4 with full persistence."""
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import klax
    import equinox as eqx
    import json
    import pickle
    from pathlib import Path
    from . import models as tm
    from . import losses as tl
    from . import data_t2 as td2
    
    (model_type, n_train, run_idx, init_idx,  # init_idx hinzugefügt
     all_C, all_I, all_F, all_W, all_P, inv_weights, G_ti,
     steps, num_hidden_layers, nodes_per_layer, batch_size, learning_rate,
     master_seed, out_dir) = args
    
    out_dir = Path(out_dir)
    
    num_paths = len(all_C)
    test_size = max(0.1, min(0.9, 1.0 - (n_train / num_paths)))
    
    # Seeds: run_idx für Datensplit, init_idx für Modell-Initialisierung
    base_seed = master_seed + run_idx * 10000 + n_train * 100
    split_key = jrandom.PRNGKey(base_seed)  # gleich für alle inits im selben run
    model_key = jrandom.PRNGKey(base_seed + init_idx * 1000 + (1 if model_type == 'FFNN' else 2))
    train_key = jrandom.PRNGKey(base_seed + init_idx * 1000 + (3 if model_type == 'FFNN' else 4))
    
    # Tag mit init
    model_id = "WITI" if model_type == "PANN" else "MSW"
    tag = f"{model_id}_n{n_train:03d}_l{num_hidden_layers}_n{nodes_per_layer}_run{run_idx:02d}_init{init_idx:02d}"
    model_path = out_dir / f"{tag}.eqx"
    history_path = out_dir / f"{tag}_history.pkl"
    meta_path = out_dir / f"{tag}.json"
    
    try:
        if model_type == 'FFNN':
            train_data, test_data, train_idx, test_idx = td2.prepare_FFNN_split(
                all_C, all_P, inv_weights, test_size=test_size, key=split_key)
            X_test, Y_test = test_data
            
            model = tm.build(key=model_key, input_dim=6, output_dim=9,
                           num_hidden_layers=num_hidden_layers,
                           nodes_per_layer=nodes_per_layer,
                           activations=jax.nn.softplus,
                           constrain_icnn_weights=False)
            trained, history = tm.train_model(model, train_data, train_key,
                                             steps=steps, batch_size=batch_size,
                                             learning_rate=learning_rate,
                                             loss_fn=tl.WeightedMSE())
            final_model = klax.finalize(trained)
            
            P_pred = jax.vmap(final_model)(X_test)
            error = float(jnp.sqrt(jnp.mean((P_pred - Y_test)**2)))
            
        else:  # PANN
            train_data, test_data, train_idx, test_idx = td2.prepare_PANN_split(
                all_F, all_I, all_W, all_P, inv_weights,
                test_size=test_size, key=split_key)
            (F_test, I_test), (W_test, P_test) = test_data
            
            model = tm.SobolevModel_WI_ti(G_ti=G_ti, key=model_key, input_dim=5,
                                       output_dim="scalar",
                                       num_hidden_layers=num_hidden_layers,
                                       nodes_per_layer=nodes_per_layer,
                                       activation=jax.nn.softplus,
                                       is_icnn=False, is_ficnn=True)
            trained, history = tm.train_WI(model, train_data, train_key,
                                          steps=steps, batch_size=batch_size,
                                          learning_rate=learning_rate,
                                          loss_fn=tl.WeightedSobolevLoss(alpha=1.0, beta=1.0))
            final_model = klax.finalize(trained)
            
            preds = jax.vmap(final_model)((F_test, I_test))
            _, P_pred = preds
            if P_pred.ndim == 3: P_pred = P_pred.reshape(-1, 9)
            if P_test.ndim == 3: P_test = P_test.reshape(-1, 9)
            error = float(jnp.sqrt(jnp.mean((P_pred - P_test)**2)))
        
        eqx.tree_serialise_leaves(str(model_path), final_model)
        
        with open(history_path, "wb") as f:
            pickle.dump(history, f)
        
        meta = {
            "task": "4",
            "model_id": model_id,
            "model_type": model_type,
            "tag": tag,
            "n_train": n_train,
            "run_idx": run_idx,
            "init_idx": init_idx,  # NEU
            "test_error": error,
            "architecture": {
                "num_hidden_layers": num_hidden_layers,
                "nodes_per_layer": nodes_per_layer,
                "activation": "softplus"
            },
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "test_size": test_size,
            "base_seed": base_seed,
            "train_idx": [int(i) for i in train_idx],
            "test_idx": [int(i) for i in test_idx],
            "saved_model_path": str(model_path),
            "saved_history_path": str(history_path),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            
    except Exception as e:
        print(f"Error: {model_type} n={n_train} run={run_idx} init={init_idx}: {e}")
        error = float('nan')
    
    return {
        'Model': 'Naive FFNN' if model_type == 'FFNN' else 'PANN',
        'N_Train_Paths': n_train,
        'Run': run_idx,
        'Init': init_idx,  # NEU
        'Test_Error': error
    }

def prepare_dataset_3(
    *,
    path_h5: str | Path,
    G_cub: jnp.ndarray,
    calibration_name_filters: tuple[str, ...] = ("uniaxial", "shear_simple"),
    test_name_filters: tuple[str, ...] = ("shear_combined",),
):
    """
    Prepare Dataset 3 (multiscale deformation paths) in a reusable form.

    Implements the logic from your Task 5 snippet:
      - load multiscale paths from .h5
      - compute alpha scaling from max |P| across all paths
      - scale P and W per path
      - compute inverse path weights per path from scaled stresses
      - split calibration vs test paths based on name filters
      - build calibration arrays + per-sample weights aligned with concatenation
      - compute cubic invariants I(F, G_cub)
      - package weighted Sobolev training data for WI_cubic

    Returns
    -------
    dict with keys:
      - mode_names, calibration_keys, test_keys
      - F_dict, P_dict, W_dict (unscaled)
      - P_scaled_dict, W_scaled_dict
      - alpha
      - inv_path_weights_dict
      - F_cal, P_cal, W_cal, weights_cal, I_cal
      - F_test, P_test, W_test, I_test
      - train_data_WI_cubic, test_data_WI_cubic
    """
    path_h5 = str(path_h5)

    # Load data separated into loadpaths
    F_paths, P_paths, W_paths, J_paths, mode_names = td2.load_multiscale_paths(path_h5)

    # Dict access by path name
    F_dict = {k: jnp.array(v) for k, v in zip(mode_names, F_paths)}
    P_dict = {k: jnp.array(v) for k, v in zip(mode_names, P_paths)}
    W_dict = {k: jnp.array(v) for k, v in zip(mode_names, W_paths)}

    # Flatten across all loadpaths to compute alpha
    P_all = jnp.concatenate([P_dict[k] for k in mode_names], axis=0)
    P_max = jnp.max(jnp.abs(P_all))
    alpha = 1.0 / P_max

    # Scale per path
    P_scaled_dict = {k: alpha * P_dict[k] for k in mode_names}
    W_scaled_dict = {k: alpha * W_dict[k] for k in mode_names}

    # Inverse path weights from *scaled* stresses
    inv_path_weights_dict = {}
    for k in mode_names:
        w_k = td2.compute_path_weight(P_scaled_dict[k])
        inv_path_weights_dict[k] = 1.0 / w_k

    # Select calibration/test keys by name filters (case-insensitive)
    def _match_any(name: str, filters: tuple[str, ...]) -> bool:
        name_l = name.lower()
        return any(f.lower() in name_l for f in filters)

    calibration_keys = [k for k in mode_names if _match_any(k, calibration_name_filters)]
    test_keys = [k for k in mode_names if _match_any(k, test_name_filters)]

    # Calibration data (scaled)
    F_cal = jnp.concatenate([F_dict[k] for k in calibration_keys], axis=0)
    P_cal = jnp.concatenate([P_scaled_dict[k] for k in calibration_keys], axis=0)
    W_cal = jnp.concatenate([W_scaled_dict[k] for k in calibration_keys], axis=0)

    # Per-sample weights aligned with the calibration concatenation order
    weights_cal = jnp.concatenate(
        [jnp.ones(P_scaled_dict[k].shape[0]) * inv_path_weights_dict[k] for k in calibration_keys],
        axis=0
    ).reshape(-1)

    # Test data (scaled)
    F_test = jnp.concatenate([F_dict[k] for k in test_keys], axis=0)
    P_test = jnp.concatenate([P_scaled_dict[k] for k in test_keys], axis=0)
    W_test = jnp.concatenate([W_scaled_dict[k] for k in test_keys], axis=0)

    # Invariants (cubic)
    I_cal = td2.compute_all_invariants_cubic(F_cal, G_cub)
    I_test = td2.compute_all_invariants_cubic(F_test, G_cub)

    # Package datasets
    train_data_WI_cubic = (
        (F_cal, I_cal),
        ((W_cal, P_cal), weights_cal)
    )
    test_data_WI_cubic = (
        (F_test, I_test),
        (W_test, P_test)
    )

    return {
        "path_h5": path_h5,
        "mode_names": mode_names,
        "calibration_keys": calibration_keys,
        "test_keys": test_keys,

        "F_dict": F_dict,
        "P_dict": P_dict,
        "W_dict": W_dict,

        "alpha": alpha,
        "P_max": P_max,

        "P_scaled_dict": P_scaled_dict,
        "W_scaled_dict": W_scaled_dict,
        "inv_path_weights_dict": inv_path_weights_dict,

        "F_cal": F_cal, "P_cal": P_cal, "W_cal": W_cal, "weights_cal": weights_cal, "I_cal": I_cal,
        "F_test": F_test, "P_test": P_test, "W_test": W_test, "I_test": I_test,

        "train_data_WI_cubic": train_data_WI_cubic,
        "test_data_WI_cubic": test_data_WI_cubic,
    }

# Functions for getting the test data
def load_run_meta(meta_path: str | Path) -> dict:
    meta_path = Path(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _reconstruct_dataset3_test_from_meta(meta: dict, *, G_cub: jnp.ndarray, tol: float = 1e-8):
    """
    Reconstruct Dataset 3 test set EXACTLY as in prepare_dataset_3, but with keys and alpha validated
    against meta.json.

    Returns:
      - For WICUB: ((F_test, I_test), (W_test, P_test))
      - For WF / WF_AUG: (F_test, (W_test, P_test))
    """
    ds3_meta = meta.get("dataset_3", {})
    path_h5 = ds3_meta.get("path_h5", None)
    if path_h5 is None:
        raise ValueError("Meta does not contain dataset_3.path_h5; cannot reconstruct Dataset 3.")

    test_keys = ds3_meta.get("test_keys", None)
    if not test_keys:
        raise ValueError("Meta does not contain dataset_3.test_keys; cannot reconstruct Dataset 3 test set.")

    alpha_ref = float(ds3_meta.get("alpha_scale", None))
    if alpha_ref is None:
        raise ValueError("Meta does not contain dataset_3.alpha_scale; cannot validate scaling.")

    # Load all paths from H5
    F_paths, P_paths, W_paths, J_paths, mode_names = td2.load_multiscale_paths(path_h5)

    # Dicts by name
    F_dict = {k: jnp.array(v) for k, v in zip(mode_names, F_paths)}
    P_dict = {k: jnp.array(v) for k, v in zip(mode_names, P_paths)}
    W_dict = {k: jnp.array(v) for k, v in zip(mode_names, W_paths)}

    # Validate that requested keys exist
    missing = [k for k in test_keys if k not in F_dict]
    if missing:
        raise ValueError(f"Meta test_keys not found in H5: {missing}")

    # Recompute alpha exactly as in prepare_dataset_3 (global max |P|)
    P_all = jnp.concatenate([P_dict[k] for k in mode_names], axis=0)
    P_max = jnp.max(jnp.abs(P_all))
    alpha = float(1.0 / P_max)

    if abs(alpha - alpha_ref) > tol:
        raise ValueError(
            f"Dataset 3 alpha mismatch. meta alpha={alpha_ref}, recomputed alpha={alpha}. "
            f"This indicates a different H5 file, a modified dataset, or non-matching preprocessing."
        )

    # Scale per path (exactly as training)
    P_scaled_dict = {k: alpha * P_dict[k] for k in mode_names}
    W_scaled_dict = {k: alpha * W_dict[k] for k in mode_names}

    # Build test arrays in the exact order stored in meta["test_keys"]
    F_test = jnp.concatenate([F_dict[k] for k in test_keys], axis=0)
    P_test = jnp.concatenate([P_scaled_dict[k] for k in test_keys], axis=0)
    W_test = jnp.concatenate([W_scaled_dict[k] for k in test_keys], axis=0)

    # Invariants for cubic WI
    I_test = td2.compute_all_invariants_cubic(F_test, G_cub)

    return {
        "F_test": F_test,
        "I_test": I_test,
        "W_test": W_test,
        "P_test": P_test,
        "alpha": alpha,
        "test_keys": test_keys,
    }


def get_test_data_for_run(
    meta_path: str | Path,
    *,
    dataset_1: dict | None = None,
    G_cub: jnp.ndarray | None = None,
):
    """
    No-brainer test-set getter.

    It uses the run's meta.json to determine:
      - which dataset the model belongs to
      - which test sets apply
      - how scaling/splitting was done (Dataset 3 models)
    and returns the test data in the correct format for that model.

    Returns
    -------
    dict[str, object]
      Keys are explicit test-set names so you cannot "accidentally pick the wrong one".
    """
    meta = load_run_meta(meta_path)
    model_id = meta.get("model_id", "").upper()

    # ------------------------
    # Dataset 1 models (Task 2/3)
    # ------------------------
    if model_id in ("MS", "MSW"):
        if dataset_1 is None:
            raise ValueError("MS/MSW require dataset_1=prepare_dataset_1(...) to guarantee correct test sets.")

        # Workflows explicitly use biax_test and mixed_test for MS/MSW. 
        X_bi_test = jax.vmap(td2.C_to_six)(dataset_1["C_bi_test"])
        Y_bi_test = dataset_1["P_bi_test"].reshape(dataset_1["P_bi_test"].shape[0], 9)

        X_mix_test = jax.vmap(td2.C_to_six)(dataset_1["C_mix_test"])
        Y_mix_test = dataset_1["P_mix_test"].reshape(dataset_1["P_mix_test"].shape[0], 9)

        return {
            "biax_test": (X_bi_test, Y_bi_test),
            "mixed_test": (X_mix_test, Y_mix_test),
        }

    if model_id == "WITI":
        if dataset_1 is None:
            raise ValueError("WITI requires dataset_1=prepare_dataset_1(...) to guarantee correct test set.")
        # In Task 3 you evaluate on mixed_test (and sometimes identity). 
        return {
            "mixed_test": ((dataset_1["F_mix_test"], dataset_1["I_mix_test"]),
                           (dataset_1["W_mix_test"], dataset_1["P_mix_test"])),
            "biax_test": ((dataset_1["F_bi_test"], dataset_1["I_bi_test"]),
                          (dataset_1["W_bi_test"], dataset_1["P_bi_test"])),
        }

    # ------------------------
    # Dataset 3 models (Task 5)
    # ------------------------
    if model_id in ("WICUB", "WF", "WF_AUG"):
        if G_cub is None:
            raise ValueError("WICUB/WF/WF_AUG require G_cub to reconstruct invariants consistently.")
        ds3 = _reconstruct_dataset3_test_from_meta(meta, G_cub=G_cub)

        if model_id == "WICUB":
            return {
                "dataset3_test": ((ds3["F_test"], ds3["I_test"]), (ds3["W_test"], ds3["P_test"])),
            }

        # WF and WF_AUG: model input is F only; test set is never augmented in Task 5.4. 
        return {
            "dataset3_test": (ds3["F_test"], (ds3["W_test"], ds3["P_test"])),
        }

    raise ValueError(f"Unknown or unsupported model_id='{model_id}' in meta: {meta_path}")


#------------------------Workflows Model Training

def workflow_task_2_2_train_ms_sweep(
    dataset_1: dict,
    *,
    out_dir: str | Path = "artifacts/task2_2",
    n_inits: int = 5,
    steps_list: list[int] | None = None,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
):
    """
    Task 2.2: Train MS(C)->P with multiple architectures, multiple training steps,
    and multiple random initializations per configuration.

    Saves per run:
      - model:   <out_dir>/MS_<arch>_l{l}_n{n}_steps{steps}_initXX.eqx
      - meta:    <out_dir>/MS_<arch>_l{l}_n{n}_steps{steps}_initXX.json
      - history: <out_dir>/MS_<arch>_l{l}_n{n}_steps{steps}_initXX_history.pkl

    Returns
    -------
    list[dict]
      One meta dict per trained run.
    """
    out_dir = _ensure_dir(out_dir)

    if steps_list is None:
        steps_list = [100_000, 300_000, 500_000, 700_000, 900_000]

    # ----- Build calibration dataset (same as your code) -----
    C_cal_MS = jnp.concatenate([dataset_1["C_uni"], dataset_1["C_ps"], dataset_1["C_bi"]], axis=0)
    P_cal_MS = jnp.concatenate([dataset_1["P_uni"], dataset_1["P_ps"], dataset_1["P_bi"]], axis=0)

    X_cal_MS = jax.vmap(td2.C_to_six)(C_cal_MS)                 # (N,6)
    Y_cal_MS = P_cal_MS.reshape(P_cal_MS.shape[0], 9)           # (N,9)
    train_data_MS = (X_cal_MS, Y_cal_MS)

    # ----- Test sets -----
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

    # Key pool (deterministic)
    master_key = dataset_1["master_key"]
    total_runs = len(archs) * len(steps_list) * n_inits
    keys = jrandom.split(master_key, total_runs * 2 + 1)
    key_cursor = 1

    results = []

    for arch_name, l, n in archs:
        for steps in steps_list:
            for init_idx in range(n_inits):
                model_key = keys[key_cursor]
                train_key = keys[key_cursor + 1]
                key_cursor += 2

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
                    loss_fn=tl.MSE(),
                )

                MS_final = klax.finalize(MS_trained)

                # Evaluate RMSE on both test sets
                P_bi_pred  = jax.vmap(MS_final)(X_bi_test)
                P_mix_pred = jax.vmap(MS_final)(X_mix_test)

                rmse_bi  = float(jnp.sqrt(jnp.mean((P_bi_pred  - Y_bi_test)  ** 2)))
                rmse_mix = float(jnp.sqrt(jnp.mean((P_mix_pred - Y_mix_test) ** 2)))

                # Save artifacts
                tag = f"MS_{arch_name}_l{l}_n{n}_steps{steps}_init{init_idx+1:02d}"
                model_path   = out_dir / f"{tag}.eqx"
                meta_path    = out_dir / f"{tag}.json"
                history_path = out_dir / f"{tag}_history.pkl"

                _save_eqx_model(MS_final, model_path)
                _save_history(MS_history, history_path)

                meta = {
                    "task": "2.2",
                    "model_id": "MS",
                    "tag": tag,
                    "arch_name": arch_name,
                    "num_hidden_layers": l,
                    "nodes_per_layer": n,
                    "activation": "softplus",
                    "steps": steps,
                    "init_idx": int(init_idx + 1),
                    "n_inits": int(n_inits),
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
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

def workflow_task_3_sweep_wi_ti_arch_steps(
    dataset_1: dict,
    *,
    strategy: str = "C",               # choose best from section 1: "A", "B", or "C"
    out_dir: str | Path = "artifacts/task3_section2",
    n_inits: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    archs=None,
    steps_list=None,
):
    """
    Task 3 (Section 2): Sweep WI_ti settings (architecture sizes + training steps)
    using ONE chosen loss strategy (A/B/C) and the loss-weighted strategy.

    Strategy meanings (WeightedSobolevLoss):
      A: alpha=1, beta=0
      B: alpha=0, beta=1
      C: alpha=1, beta=1

    Architectures:
      default: small=(l=2,n=8), medium=(l=3,n=16), large=(l=4,n=32)

    Steps:
      default: [100k, 300k, 500k]

    Each configuration is trained with n_inits random initializations.

    Saves per run:
      WITI_{strategy}_{arch}_l{l}_n{n}_steps{steps}_initXX.eqx
      WITI_{strategy}_{arch}_l{l}_n{n}_steps{steps}_initXX_history.pkl
      WITI_{strategy}_{arch}_l{l}_n{n}_steps{steps}_initXX.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy = strategy.upper()
    assert strategy in ("A", "B", "C"), "strategy must be 'A', 'B', or 'C'"

    if archs is None:
        archs = [
            ("small",  2,  8),
            ("medium", 3, 16),
            ("large",  4, 32),
        ]

    if steps_list is None:
        steps_list = [100_000, 300_000, 500_000]

    G_ti = dataset_1["G_ti"]
    master_key = dataset_1["master_key"]

    # ------------------------------------------------------------
    # Calibration data (same ordering as Task 3 Section 1):
    # IMPORTANT: order [biaxial, uniaxial, pure_shear]
    # ------------------------------------------------------------
    F_bi  = dataset_1["F_bi"]
    F_uni = dataset_1["F_uni"]
    F_ps  = dataset_1["F_ps"]

    P_bi  = dataset_1["P_bi"]
    P_uni = dataset_1["P_uni"]
    P_ps  = dataset_1["P_ps"]

    W_bi  = dataset_1["W_bi"]
    W_uni = dataset_1["W_uni"]
    W_ps  = dataset_1["W_ps"]

    I_bi  = dataset_1["I_bi"]
    I_uni = dataset_1["I_uni"]
    I_ps  = dataset_1["I_ps"]

    F_cal_all = jnp.concatenate([F_bi, F_uni, F_ps], axis=0)
    I_cal_all = jnp.concatenate([I_bi, I_uni, I_ps], axis=0)
    W_cal_all = jnp.concatenate([W_bi, W_uni, W_ps], axis=0)
    P_cal_all = jnp.concatenate([P_bi, P_uni, P_ps], axis=0)

    # ------------------------------------------------------------
    # Loss-weighting: inverse path weights
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

    sample_weights = jnp.concatenate([weights_bi, weights_uni, weights_ps], axis=0).reshape(-1)

    # Training data format for WeightedSobolevLoss:
    train_data = (
        (F_cal_all, I_cal_all),
        ((W_cal_all, P_cal_all), sample_weights)
    )

    # ------------------------------------------------------------
    # Choose loss function based on strategy
    # ------------------------------------------------------------
    if strategy == "A":
        loss_fn = tl.WeightedSobolevLoss(alpha=1.0, beta=0.0)
    elif strategy == "B":
        loss_fn = tl.WeightedSobolevLoss(alpha=0.0, beta=1.0)
    else:  # "C"
        loss_fn = tl.WeightedSobolevLoss(alpha=1.0, beta=1.0)

    # ------------------------------------------------------------
    # Train sweep: arch × steps × init
    # ------------------------------------------------------------
    results = []
    # allocate a large pool of keys
    total_runs = len(archs) * len(steps_list) * n_inits
    keys = jrandom.split(master_key, total_runs * 2 + 1)
    key_cursor = 1

    for arch_name, l, n in archs:
        for steps in steps_list:
            for init_idx in range(n_inits):
                model_key = keys[key_cursor]
                train_key = keys[key_cursor + 1]
                key_cursor += 2

                # Build WI_ti model with current architecture
                WI_ti_model = tm.SobolevModel_WI_ti(
                    G_ti=G_ti,
                    key=model_key,
                    input_dim=5,
                    output_dim="scalar",
                    num_hidden_layers=l,
                    nodes_per_layer=n,
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

                tag = f"WITI_{strategy}_{arch_name}_l{l}_n{n}_steps{steps}_init{init_idx+1:02d}"
                model_path   = out_dir / f"{tag}.eqx"
                history_path = out_dir / f"{tag}_history.pkl"
                meta_path    = out_dir / f"{tag}.json"

                eqx.tree_serialise_leaves(str(model_path), final_model)
                with open(history_path, "wb") as f:
                    pickle.dump(history, f)

                meta = {
                    "task": "3_section2",
                    "model_id": "WITI",
                    "strategy": strategy,
                    "tag": tag,
                    "num_hidden_layers": l,
                    "nodes_per_layer": n,
                    "activation": "softplus",
                    "steps": steps,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "loss": "WeightedSobolevLoss",
                    "loss_params": {"alpha": float(loss_fn.alpha), "beta": float(loss_fn.beta)},
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

def workflow_task_3_calibration_set_study(
    dataset_1: dict,
    *,
    # PLACEHOLDERS: you will set these after inspecting previous results
    best_l: int = 3,
    best_n: int = 16,
    best_strategy: str = "C",          # "A" | "B" | "C"
    steps: int = 300_000,

    out_dir: str | Path = "artifacts/task3_calibration_study",
    n_inits: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,

    # which calibration paths are available:
    # you can extend this dict later (e.g. add a 4th path) without changing the workflow
    include_single_paths: bool = True,
    include_pair_paths: bool = True,
    include_all_paths: bool = True,
):
    """
    Task 3 (final section): Train WI_ti on different calibration subsets (loadpath combinations)
    using the best architecture + best loss strategy determined earlier.

    - Uses loss-weighted strategy for each subset:
        weight per sample = 1 / compute_path_weight(P_path)
      and concatenates weights consistent with concatenation order.

    - Trains n_inits random initializations per calibration subset.

    Saves per run:
      WITI_CAL_<subset>_<strategy>_l{l}_n{n}_steps{steps}_initXX.eqx
      WITI_CAL_<subset>_<strategy>_l{l}_n{n}_steps{steps}_initXX_history.pkl
      WITI_CAL_<subset>_<strategy>_l{l}_n{n}_steps{steps}_initXX.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_strategy = best_strategy.upper()
    assert best_strategy in ("A", "B", "C"), "best_strategy must be 'A', 'B', or 'C'"

    # Choose loss function
    if best_strategy == "A":
        loss_fn = tl.WeightedSobolevLoss(alpha=1.0, beta=0.0)
    elif best_strategy == "B":
        loss_fn = tl.WeightedSobolevLoss(alpha=0.0, beta=1.0)
    else:
        loss_fn = tl.WeightedSobolevLoss(alpha=1.0, beta=1.0)

    G_ti = dataset_1["G_ti"]
    master_key = dataset_1["master_key"]

    # ------------------------------------------------------------------
    # Available calibration paths (Dataset 1 calibration)
    # NOTE: you can add a 4th calibration path here later if you have it.
    # Each entry provides F, I, W, P for that path.
    # ------------------------------------------------------------------
    cal_paths = {
        "biaxial": {
            "F": dataset_1["F_bi"],
            "I": dataset_1["I_bi"],
            "W": dataset_1["W_bi"],
            "P": dataset_1["P_bi"],
        },
        "uniaxial": {
            "F": dataset_1["F_uni"],
            "I": dataset_1["I_uni"],
            "W": dataset_1["W_uni"],
            "P": dataset_1["P_uni"],
        },
        "pure_shear": {
            "F": dataset_1["F_ps"],
            "I": dataset_1["I_ps"],
            "W": dataset_1["W_ps"],
            "P": dataset_1["P_ps"],
        },
    }

    path_names = list(cal_paths.keys())

    # ------------------------------------------------------------------
    # Build list of calibration subsets
    # - singletons:  (biaxial), (uniaxial), (pure_shear)
    # - pairs:       (biaxial+uniaxial), ...
    # - all:         (biaxial+uniaxial+pure_shear)
    # ------------------------------------------------------------------
    subsets = []
    if include_single_paths:
        subsets += [(p,) for p in path_names]
    if include_pair_paths:
        subsets += list(itertools.combinations(path_names, 2))
    if include_all_paths:
        subsets += [tuple(path_names)]

    # ------------------------------------------------------------------
    # Helper: build weighted training data for a given subset
    # ------------------------------------------------------------------
    def build_train_data_for_subset(subset):
        # Concatenate in the subset order (subset tuple order)
        F_all = jnp.concatenate([cal_paths[p]["F"] for p in subset], axis=0)
        I_all = jnp.concatenate([cal_paths[p]["I"] for p in subset], axis=0)
        W_all = jnp.concatenate([cal_paths[p]["W"] for p in subset], axis=0)
        P_all = jnp.concatenate([cal_paths[p]["P"] for p in subset], axis=0)

        # Compute per-path weights, then expand to per-sample weights
        weights_list = []
        weights_meta = {}
        for p in subset:
            P_path = cal_paths[p]["P"]
            w_path = td2.compute_path_weight(P_path)
            w_inv = 1.0 / w_path
            weights_list.append(w_inv * jnp.ones(P_path.shape[0]))
            weights_meta[p] = {"w": float(w_path), "w_inv": float(w_inv), "n_samples": int(P_path.shape[0])}

        sample_weights = jnp.concatenate(weights_list, axis=0).reshape(-1)

        train_data = (
            (F_all, I_all),
            ((W_all, P_all), sample_weights)
        )
        return train_data, weights_meta

    # ------------------------------------------------------------------
    # Train all subsets × n_inits
    # ------------------------------------------------------------------
    results = []
    total_runs = len(subsets) * n_inits
    keys = jrandom.split(master_key, total_runs * 2 + 1)
    key_cursor = 1

    for subset in subsets:
        subset_tag = "+".join(subset)  # e.g. "biaxial+uniaxial"
        train_data, weights_meta = build_train_data_for_subset(subset)

        for init_idx in range(n_inits):
            model_key = keys[key_cursor]
            train_key = keys[key_cursor + 1]
            key_cursor += 2

            # Build best-architecture WI_ti model
            model = tm.SobolevModel_WI_ti(
                G_ti=G_ti,
                key=model_key,
                input_dim=5,
                output_dim="scalar",
                num_hidden_layers=best_l,
                nodes_per_layer=best_n,
                activation=jax.nn.softplus,
                is_icnn=False,
                is_ficnn=True
            )

            trained_model, history = tm.train_WI(
                model=model,
                train_data=train_data,
                key=train_key,
                steps=steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                loss_fn=loss_fn
            )

            final_model = klax.finalize(trained_model)

            tag = (
                f"WITI_CAL_{subset_tag}_{best_strategy}"
                f"_l{best_l}_n{best_n}_steps{steps}"
                f"_init{init_idx+1:02d}"
            )

            model_path   = out_dir / f"{tag}.eqx"
            history_path = out_dir / f"{tag}_history.pkl"
            meta_path    = out_dir / f"{tag}.json"

            eqx.tree_serialise_leaves(str(model_path), final_model)
            with open(history_path, "wb") as f:
                pickle.dump(history, f)

            meta = {
                "task": "3_calibration_study",
                "model_id": "WITI",
                "tag": tag,
                "subset": list(subset),
                "subset_tag": subset_tag,
                "architecture": {"l": best_l, "n": best_n, "activation": "softplus"},
                "strategy": best_strategy,
                "loss": "WeightedSobolevLoss",
                "loss_params": {"alpha": float(loss_fn.alpha), "beta": float(loss_fn.beta)},
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "loss_weighting": "inverse path weights computed on subset paths",
                "weights_by_path": weights_meta,
                "saved_model_path": str(model_path),
                "saved_history_path": str(history_path),
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            results.append(meta)

    return results

def workflow_task_4_generalization(
    ds4: dict,
    *,
    out_dir: str | Path = "artifacts/task4",
    n_paths_list: list[int] = [1, 2, 4, 8, 16, 32, 48, 64, 80],
    n_runs: int = 5,
    n_inits: int = 1,  # NEU
    steps: int = 100_000,
    num_hidden_layers: int = 3,
    nodes_per_layer: int = 16,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    n_jobs: int = -2,
    master_seed: int = 42,
):
    """Task 4: Generalization experiment FFNN vs PANN with full persistence."""
    from joblib import Parallel, delayed
    import pandas as pd
    
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    
    experiments = []
    for run_idx in range(n_runs):
        for init_idx in range(n_inits):  # NEU
            for n_train in n_paths_list:
                for model_type in ['FFNN', 'PANN']:
                    experiments.append((
                        model_type, n_train, run_idx, init_idx,  # init_idx hinzugefügt
                        ds4["all_C"], ds4["all_I"], ds4["all_F"],
                        ds4["all_W"], ds4["all_P"], ds4["inv_weights"], ds4["G_ti"],
                        steps, num_hidden_layers, nodes_per_layer, batch_size, learning_rate,
                        master_seed, str(out_dir)
                    ))
    
    total = len(experiments)
    print(f"Task 4: {len(n_paths_list)} sizes × {n_runs} runs × {n_inits} inits × 2 models = {total} trainings")
    
    results = Parallel(n_jobs=n_jobs, verbose=total, backend='loky')(
        delayed(_run_single_task4)(exp) for exp in experiments
    )
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / f"results_summary_l{num_hidden_layers}_n{nodes_per_layer}_steps{steps}.csv", index=False)
    
    return {"results_df": results_df, "out_dir": str(out_dir)}

def workflow_task_5_2_sweep_wi_cubic(
    dataset_3: dict,
    *,
    G_cub: jnp.ndarray,
    out_dir: str | Path = "artifacts/task5_2",
    n_inits: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    steps_list=None,
    archs=None,
    # default: energy-only like your snippet; override if you want later
    loss_alpha: float = 1.0,
    loss_beta: float = 0.0,
    master_key: jrandom.PRNGKey = jrandom.PRNGKey(0),
):
    """
    Task 5.2: Sweep WI_cubic model configurations on Dataset 3.

    Model: SobolevModel_WI_Cubic (W + grad(P) via Sobolev), trained with WeightedSobolevLoss.

    Default sweep:
      archs: small=(l=2,n=8), medium=(l=3,n=16), large=(l=4,n=32)
      steps: [100k, 300k, 500k]
      n_inits: 5

    Training data is taken from dataset_3["train_data_WI_cubic"] which already includes weights.

    Saves:
      WICUB_<strategy>_<arch>_l{l}_n{n}_steps{steps}_initXX.eqx
      ..._history.pkl
      ...json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if steps_list is None:
        steps_list = [100_000, 300_000, 500_000]

    if archs is None:
        archs = [
            ("small",  2,  8),
            ("medium", 3, 16),
            ("large",  4, 32),
        ]

    train_data = dataset_3["train_data_WI_cubic"]
    test_data = dataset_3["test_data_WI_cubic"]

    loss_fn = tl.WeightedSobolevLoss(alpha=loss_alpha, beta=loss_beta)
    strategy_tag = f"a{loss_alpha:g}_b{loss_beta:g}"  # e.g. a1_b0

    # keys for all runs
    total_runs = len(archs) * len(steps_list) * n_inits
    keys = jrandom.split(master_key, total_runs * 2 + 1)
    key_cursor = 1

    results = []

    for arch_name, l, n in archs:
        for steps in steps_list:
            for init_idx in range(n_inits):
                model_key = keys[key_cursor]
                train_key = keys[key_cursor + 1]
                key_cursor += 2

                # Build WI_cubic model
                WI_cubic_model = tm.SobolevModel_WI_Cubic(
                    G_cub=G_cub,
                    key=model_key,
                    input_dim=6,          # cubic invariants dim
                    output_dim="scalar",
                    num_hidden_layers=l,
                    nodes_per_layer=n,
                    activation=jax.nn.softplus,
                    is_icnn=False,
                    is_ficnn=True
                )

                trained_model, history = tm.train_model(
                    WI_cubic_model,
                    train_data,
                    train_key,
                    steps=steps,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    loss_fn=loss_fn
                )

                final_model = klax.finalize(trained_model)

                tag = f"WICUB_{strategy_tag}_{arch_name}_l{l}_n{n}_steps{steps}_init{init_idx+1:02d}"
                model_path = out_dir / f"{tag}.eqx"
                hist_path  = out_dir / f"{tag}_history.pkl"
                meta_path  = out_dir / f"{tag}.json"

                # Use your existing helpers from earlier workflows:
                _save_eqx_model(final_model, model_path)
                _save_history(history, hist_path)

                meta = {
                    "task": "5.2",
                    "model_id": "WICUB",
                    "tag": tag,
                    "architecture": {"l": l, "n": n, "activation": "softplus"},
                    "steps": steps,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "loss": "WeightedSobolevLoss",
                    "loss_params": {"alpha": float(loss_alpha), "beta": float(loss_beta)},
                    "dataset_3": {
                        "path_h5": dataset_3.get("path_h5", None),
                        "alpha_scale": float(dataset_3["alpha"]),
                        "P_max": float(dataset_3["P_max"]),
                        "n_cal_paths": len(dataset_3["calibration_keys"]),
                        "n_test_paths": len(dataset_3["test_keys"]),
                        "calibration_keys": dataset_3["calibration_keys"],
                        "test_keys": dataset_3["test_keys"],
                    },
                    "saved_model_path": str(model_path),
                    "saved_history_path": str(hist_path),
                }

                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                results.append(meta)

    return results

def workflow_task_5_3_sweep_wf(
    dataset_3: dict,
    *,
    out_dir: str | Path = "artifacts/task5_3",
    n_inits: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    steps_list=None,
    archs=None,
    # default: energy-only like your snippet; keep configurable
    loss_alpha: float = 1.0,
    loss_beta: float = 0.0,
    master_key: jrandom.PRNGKey = jrandom.PRNGKey(0),
):
    """
    Task 5.3: Sweep WF model configurations on Dataset 3 (training only).

    Model: SobolevModel_WF
      - Input to model call is F only.
      - Model internally constructs (F, cofF, detF) (R^19).

    Data:
      - uses dataset_3["F_cal"], dataset_3["W_cal"], dataset_3["P_cal"], dataset_3["weights_cal"]
      - uses dataset_3 scaling (alpha) and per-sample weights computed from scaled stresses

    Sweep defaults:
      archs: small=(l=2,n=8), medium=(l=3,n=16), large=(l=4,n=32)
      steps: [100k, 300k, 500k]
      n_inits: 5

    Saves per run:
      WF_{strategy}_{arch}_l{l}_n{n}_steps{steps}_initXX.eqx
      WF_{strategy}_{arch}_l{l}_n{n}_steps{steps}_initXX_history.pkl
      WF_{strategy}_{arch}_l{l}_n{n}_steps{steps}_initXX.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if steps_list is None:
        steps_list = [100_000, 300_000, 500_000]

    if archs is None:
        archs = [
            ("small",  2,  8),
            ("medium", 3, 16),
            ("large",  4, 32),
        ]

    # Unpack dataset_3 (already scaled and properly aligned)
    F_cal = dataset_3["F_cal"]
    W_cal = dataset_3["W_cal"]
    P_cal = dataset_3["P_cal"]
    weights_cal = dataset_3["weights_cal"]

    # Training data format for WeightedSobolevLoss:
    # batch = (F, ((W_true, P_true), weights))
    train_data_WF = (
        F_cal,
        ((W_cal, P_cal), weights_cal)
    )

    # Loss
    loss_fn = tl.WeightedSobolevLoss(alpha=loss_alpha, beta=loss_beta)
    strategy_tag = f"a{loss_alpha:g}_b{loss_beta:g}"  # e.g. a1_b0

    # Key pool
    total_runs = len(archs) * len(steps_list) * n_inits
    keys = jrandom.split(master_key, total_runs * 2 + 1)
    key_cursor = 1

    results = []

    for arch_name, l, n in archs:
        for steps in steps_list:
            for init_idx in range(n_inits):
                model_key = keys[key_cursor]
                train_key = keys[key_cursor + 1]
                key_cursor += 2

                # WF model (F-only input; internal feature construction)
                WF_model = tm.SobolevModel_WF(
                    key=model_key,
                    input_dim=19,            # internal (F, cofF, detF)
                    output_dim="scalar",
                    num_hidden_layers=l,
                    nodes_per_layer=n,
                    activation=jax.nn.softplus,
                    is_icnn=True,            # as in your code
                    is_ficnn=False
                )

                trained_model, history = tm.train_model(
                    model=WF_model,
                    train_data=train_data_WF,
                    key=train_key,
                    steps=steps,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    loss_fn=loss_fn
                )

                final_model = klax.finalize(trained_model)

                tag = f"WF_{strategy_tag}_{arch_name}_l{l}_n{n}_steps{steps}_init{init_idx+1:02d}"
                model_path = out_dir / f"{tag}.eqx"
                hist_path  = out_dir / f"{tag}_history.pkl"
                meta_path  = out_dir / f"{tag}.json"

                _save_eqx_model(final_model, model_path)
                _save_history(history, hist_path)

                meta = {
                    "task": "5.3",
                    "model_id": "WF",
                    "tag": tag,
                    "architecture": {"l": l, "n": n, "activation": "softplus"},
                    "steps": steps,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "loss": "WeightedSobolevLoss",
                    "loss_params": {"alpha": float(loss_alpha), "beta": float(loss_beta)},
                    "icnn": {"is_icnn": True, "is_ficnn": False},
                    "dataset_3": {
                        "path_h5": dataset_3.get("path_h5", None),
                        "alpha_scale": float(dataset_3["alpha"]),
                        "P_max": float(dataset_3["P_max"]),
                        "n_cal_paths": len(dataset_3["calibration_keys"]),
                        "n_test_paths": len(dataset_3["test_keys"]),
                        "calibration_keys": dataset_3["calibration_keys"],
                        "test_keys": dataset_3["test_keys"],
                    },
                    "saved_model_path": str(model_path),
                    "saved_history_path": str(hist_path),
                }

                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                results.append(meta)

    return results

# def workflow_task_5_4_train_wf_augmented(
#     dataset_3: dict,
#     *,
#     # Best architecture from Task 5.3 (you will set these after selection)
#     best_l: int = 3,
#     best_n: int = 16,
#     steps: int = 300_000,

#     # Augmentation settings
#     observers_list=(8, 16, 32, 64),

#     # Repeats / optimization
#     n_inits: int = 5,
#     batch_size: int = 32,
#     learning_rate: float = 1e-3,

#     # Loss (default energy-only, matches your snippet)
#     loss_alpha: float = 1.0,
#     loss_beta: float = 0.0,

#     # Saving
#     out_dir: str | Path = "artifacts/task5_4",

#     # Keys
#     master_key: jrandom.PRNGKey = jrandom.PRNGKey(0),
#     aug_key_seed: int = 1234,
# ):
#     """
#     Task 5.4: Train WF model on objectivity-augmented datasets with different numbers of observers.

#     For each num_observers in observers_list:
#       1) Augment (F_cal, W_cal, P_cal) using td2.augment_WF_data.
#       2) Duplicate per-sample weights consistent with augmentation layout:
#            augment_WF_data returns [original N] + [rotated num_obs*N]
#          -> weights_aug = concat([weights_cal, tile(weights_cal, num_obs)])
#       3) Train n_inits random initializations of the WF model with the selected architecture.
#       4) Save model + history + metadata.

#     Naming:
#       WF_AUG_obs{num_obs}_l{l}_n{n}_steps{steps}_initXX.eqx
#       WF_AUG_obs{num_obs}_l{l}_n{n}_steps{steps}_initXX_history.pkl
#       WF_AUG_obs{num_obs}_l{l}_n{n}_steps{steps}_initXX.json
#     """
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Unpack (already scaled + weights aligned to F_cal/W_cal/P_cal)
#     F_cal = dataset_3["F_cal"]
#     W_cal = dataset_3["W_cal"]
#     P_cal = dataset_3["P_cal"]
#     weights_cal = dataset_3["weights_cal"].reshape(-1)

#     loss_fn = tl.WeightedSobolevLoss(alpha=loss_alpha, beta=loss_beta)
#     strategy_tag = f"a{loss_alpha:g}_b{loss_beta:g}"

#     results = []

#     # Keys: need (model_key, train_key) per run
#     total_runs = len(observers_list) * n_inits
#     keys = jrandom.split(master_key, total_runs * 2 + 1)
#     key_cursor = 1

#     for num_obs in observers_list:
#         # -----------------------------------------
#         # 1) Data augmentation (objectivity-based)
#         # augment_WF_data returns:
#         #   F_aug: (N + num_obs*N, 3, 3)
#         #   W_aug: (N + num_obs*N,)
#         #   P_aug: (N + num_obs*N, 3, 3)
#         # with ordering [original, rotated-block]
#         # -----------------------------------------
#         aug_key = jrandom.PRNGKey(aug_key_seed + int(num_obs))
#         F_aug, W_aug, P_aug = td2.augment_WF_data(
#             F_cal, W_cal, P_cal,
#             num_observers=int(num_obs),
#             key=aug_key
#         )

#         # -----------------------------------------
#         # 2) Duplicate weights consistently
#         # weights_cal corresponds to original (N,)
#         # Rotated block repeats each original sample once per observer => tile(weights_cal, num_obs)
#         # Final size: (N + num_obs*N,) = (1+num_obs)*N
#         # -----------------------------------------
#         weights_aug = jnp.concatenate(
#             [weights_cal, jnp.tile(weights_cal, (int(num_obs),))],
#             axis=0
#         ).reshape(-1)

#         # Sanity check (safe; can remove later)
#         # assert F_aug.shape[0] == weights_aug.shape[0] == W_aug.shape[0] == P_aug.shape[0]

#         train_data_WF_aug = (
#             F_aug,
#             ((W_aug, P_aug), weights_aug)
#         )

#         # -----------------------------------------
#         # 3) Train n_inits models for this augmented dataset
#         # -----------------------------------------
#         for init_idx in range(n_inits):
#             model_key = keys[key_cursor]
#             train_key = keys[key_cursor + 1]
#             key_cursor += 2

#             WF_model = tm.SobolevModel_WF(
#                 key=model_key,
#                 input_dim=19,               # internal (F, cofF, detF)
#                 output_dim="scalar",
#                 num_hidden_layers=best_l,
#                 nodes_per_layer=best_n,
#                 activation=jax.nn.softplus,
#                 is_icnn=True,
#                 is_ficnn=False
#             )

#             trained_model, history = tm.train_model(
#                 model=WF_model,
#                 train_data=train_data_WF_aug,
#                 key=train_key,
#                 steps=steps,
#                 batch_size=batch_size,
#                 learning_rate=learning_rate,
#                 loss_fn=loss_fn
#             )

#             final_model = klax.finalize(trained_model)

#             tag = f"WF_AUG_obs{int(num_obs)}_{strategy_tag}_l{best_l}_n{best_n}_steps{steps}_init{init_idx+1:02d}"
#             model_path = out_dir / f"{tag}.eqx"
#             hist_path  = out_dir / f"{tag}_history.pkl"
#             meta_path  = out_dir / f"{tag}.json"

#             _save_eqx_model(final_model, model_path)
#             _save_history(history, hist_path)

#             meta = {
#                 "task": "5.4",
#                 "model_id": "WF_AUG",
#                 "tag": tag,
#                 "num_observers": int(num_obs),
#                 "augmentation": "td2.augment_WF_data: F->QF, P->QP, W unchanged; concatenates [orig, rotated]",  # see data_t2.py
#                 "architecture": {"l": best_l, "n": best_n, "activation": "softplus"},
#                 "steps": steps,
#                 "batch_size": batch_size,
#                 "learning_rate": learning_rate,
#                 "loss": "WeightedSobolevLoss",
#                 "loss_params": {"alpha": float(loss_alpha), "beta": float(loss_beta)},
#                 "icnn": {"is_icnn": True, "is_ficnn": False},
#                 "dataset_3": {
#                     "path_h5": dataset_3.get("path_h5", None),
#                     "alpha_scale": float(dataset_3["alpha"]),
#                     "P_max": float(dataset_3["P_max"]),
#                     "calibration_keys": dataset_3["calibration_keys"],
#                     "test_keys": dataset_3["test_keys"],
#                     "N_cal_original": int(F_cal.shape[0]),
#                     "N_cal_augmented": int(F_aug.shape[0]),
#                 },
#                 "saved_model_path": str(model_path),
#                 "saved_history_path": str(hist_path),
#             }

#             with open(meta_path, "w", encoding="utf-8") as f:
#                 json.dump(meta, f, indent=2)

#             results.append(meta)

#     return results


#------------------------Workflows Model Evaluation

def get_test_data_for_model_id(
    model_id: str,
    *,
    dataset_1: dict | None = None,
    dataset_3: dict | None = None,
    include_identity: bool = True,
):
    """
    Return the correct test dataset(s) in the correct format for a given model_id.

    Parameters
    ----------
    model_id:
        One of: "MS", "MSW", "WITI", "WICUB", "WF", "WF_AUG"
        (based on your current workflows.py)

    dataset_1:
        Output of prepare_dataset_1(). Required for Dataset 1 models:
          - MS, MSW, WITI

    dataset_3:
        Output of prepare_dataset_3(). Required for Dataset 3 models:
          - WICUB, WF, WF_AUG

    include_identity:
        If True and model_id supports it, include an identity test case.

    Returns
    -------
    dict[str, tuple]:
        A dictionary mapping test-set name -> test_data tuple in the expected format.

        Formats:
          - MS / MSW:
              test_data = (X_test, Y_test)  where
                X_test: (N,6)  = C_to_six(C(F))
                Y_test: (N,9)  = vec(P)

          - WITI:
              test_data = ((F_test, I_test), (W_test, P_test))
              (no weights in test)

          - WICUB:
              test_data = ((F_test, I_test), (W_test, P_test))
              (already prepared by prepare_dataset_3)

          - WF / WF_AUG:
              test_data = (F_test, (W_test, P_test))
              (model input is F only)
    """
    model_id = model_id.upper()

    out = {}

    # ----------------------------
    # Dataset 1 models
    # ----------------------------
    if model_id in ("MS", "MSW", "WITI"):
        if dataset_1 is None:
            raise ValueError(f"model_id='{model_id}' requires dataset_1=prepare_dataset_1(...)")

    # MS / MSW: C->six as input, P reshaped to (N,9)
    if model_id in ("MS", "MSW"):
        # biax_test
        X_bi_test = jax.vmap(td2.C_to_six)(dataset_1["C_bi_test"])
        Y_bi_test = dataset_1["P_bi_test"].reshape(dataset_1["P_bi_test"].shape[0], 9)
        out["biax_test"] = (X_bi_test, Y_bi_test)

        # mixed_test
        X_mix_test = jax.vmap(td2.C_to_six)(dataset_1["C_mix_test"])
        Y_mix_test = dataset_1["P_mix_test"].reshape(dataset_1["P_mix_test"].shape[0], 9)
        out["mixed_test"] = (X_mix_test, Y_mix_test)

        return out

    # WITI: invariant-based model, test format ((F,I),(W,P))
    if model_id == "WITI":
        # mixed_test (main task-3 test)
        out["mixed_test"] = (
            (dataset_1["F_mix_test"], dataset_1["I_mix_test"]),
            (dataset_1["W_mix_test"], dataset_1["P_mix_test"])
        )

        # optional additional test set: biax_test
        out["biax_test"] = (
            (dataset_1["F_bi_test"], dataset_1["I_bi_test"]),
            (dataset_1["W_bi_test"], dataset_1["P_bi_test"])
        )

        if include_identity:
            G_ti = dataset_1["G_ti"]
            F_I = jnp.eye(3)[None, :, :]          # (1,3,3)
            I_I = td2.compute_all_invariants(F_I, G_ti)
            # you may or may not have analytical references available; keep targets optional
            out["identity_input_only"] = ((F_I, I_I), None)

        return out

    # ----------------------------
    # Dataset 3 models
    # ----------------------------
    if model_id in ("WICUB", "WF", "WF_AUG"):
        if dataset_3 is None:
            raise ValueError(f"model_id='{model_id}' requires dataset_3=prepare_dataset_3(...)")

    # WICUB: already packaged by prepare_dataset_3
    if model_id == "WICUB":
        out["dataset3_test"] = dataset_3["test_data_WI_cubic"]
        if include_identity:
            # cubic identity: build (F,I) input only
            F_I = jnp.eye(3)[None, :, :]
            # NOTE: prepare_dataset_3 does not store G_cub; caller should re-use same G_cub they used.
            # If you want identity for cubic invariants, pass it externally or store G_cub in dataset_3.
            out["identity_input_only_note"] = "To add identity for WICUB, store G_cub in dataset_3 or pass it in."
        return out

    # WF / WF_AUG: F-only input, test format (F,(W,P))
    if model_id in ("WF", "WF_AUG"):
        out["dataset3_test"] = (
            dataset_3["F_test"],
            (dataset_3["W_test"], dataset_3["P_test"])
        )
        if include_identity:
            F_I = jnp.eye(3)[None, :, :]
            out["identity_input_only"] = (F_I, None)
        return out

    raise ValueError(
        f"Unknown model_id='{model_id}'. Expected one of: "
        f"MS, MSW, WITI, WICUB, WF, WF_AUG"
    )

# ------------------------------------------------------------
# Generic model loader
# ------------------------------------------------------------



def load_saved_models(
    artifacts_dir: str | Path,
    *,
    # Filter: only load certain model families, e.g. "MS", "MSW", "WITI", "WICUB", "WF", "WF_AUG"
    model_id: str | None = None,
    # Filter: for Task 4 (meta["model_type"] is "FFNN" or "PANN")
    model_type: str | None = None,
    # Optional: restrict to tags containing these substrings (AND semantics)
    tag_contains: str | Iterable[str] | None = None,
    # Required only for WI TI models (WITI / PANN in your repo)
    dataset_1: dict | None = None,
    # Required only for WICUB models
    G_cub=None,
    # Deterministic key to build "like" structures
    like_key: jrandom.PRNGKey = jrandom.PRNGKey(0),
    # Strictness: if True, missing eqx/meta raises; else skips
    strict: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Load saved EQX models from a directory of artifacts produced by workflows.py.

    Returns
    -------
    dict[tag, {"model": <equinox.Module>, "meta": dict, "eqx_path": Path, "meta_path": Path}]
    """

    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"artifacts_dir does not exist: {artifacts_dir}")

    # Normalize filters
    model_id_u = model_id.upper() if model_id else None
    model_type_u = model_type.upper() if model_type else None

    if isinstance(tag_contains, str):
        tag_contains_list = [tag_contains]
    elif tag_contains is None:
        tag_contains_list = []
    else:
        tag_contains_list = list(tag_contains)

    meta_paths = sorted(artifacts_dir.glob("*.json"))
    if not meta_paths:
        raise FileNotFoundError(f"No *.json meta files found in: {artifacts_dir}")

    loaded: dict[str, dict[str, Any]] = {}

    for mp in meta_paths:
        meta = _load_run_meta_local(mp)

        # ---- Identify model family (Task 2/3/5 uses model_id; Task 4 uses model_type) ----
        mid = str(meta.get("model_id", "")).upper().strip()
        mtype = str(meta.get("model_type", "")).upper().strip()
        tag = str(meta.get("tag", meta.get("model_name", mp.stem)))

        # Apply filters
        if model_id_u and mid != model_id_u:
            continue
        if model_type_u and mtype != model_type_u:
            continue
        if tag_contains_list and not all(s in tag for s in tag_contains_list):
            continue

        # ---- Find eqx path ----
        eqx_path = _resolve_eqx_path_from_meta(meta, meta_path=mp)

        if not eqx_path.exists():
            msg = f"EQX file not found for meta '{mp.name}': {eqx_path}"
            if strict:
                raise FileNotFoundError(msg)
            else:
                print(f"[load_saved_models] SKIP: {msg}")
                continue

        # ---- Build like-model ----
        like_model = _build_like_model_from_meta(
            meta,
            like_key=like_key,
            dataset_1=dataset_1,
            G_cub=G_cub,
        )

        # ---- IMPORTANT: WITI like-model must be finalized before deserialisation ----
        if str(meta.get("model_id", "")).upper().strip() == "WITI":
            like_model = klax.finalize(like_model)

        # ---- Deserialize ----
        model = eqx.tree_deserialise_leaves(str(eqx_path), like=like_model)

        # Optional safety net (harmless if already finalized)
        if str(meta.get("model_id", "")).upper().strip() == "WITI":
            model = klax.finalize(model)


        loaded[tag] = {
            "model": model,
            "meta": meta,
            "eqx_path": eqx_path,
            "meta_path": mp,
        }

    if not loaded and strict:
        raise FileNotFoundError(
            f"No models loaded from {artifacts_dir} with filters: "
            f"model_id={model_id_u}, model_type={model_type_u}, tag_contains={tag_contains_list}"
        )

    return loaded


# -------------------------
# Internal helpers
# -------------------------
def _load_run_meta_local(meta_path: str | Path) -> dict[str, Any]:
    meta_path = Path(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_eqx_path_from_meta(meta: dict[str, Any], *, meta_path: Path) -> Path:
    """
    Prefer meta["saved_model_path"]. If absent, infer {meta_path.stem}.eqx in same folder.
    """
    p = meta.get("saved_model_path", None)
    if p:
        return Path(p)
    # Fallback to same-stem .eqx next to meta
    return meta_path.with_suffix(".eqx")


import re

def _get_arch_from_meta(meta: dict[str, Any]) -> tuple[int, int]:
    """
    Supported sources (in priority order):
      1) meta["num_hidden_layers"], meta["nodes_per_layer"]
      2) meta["architecture"]["l"], meta["architecture"]["n"]
      3) meta["benchmark_architecture"]["l"], meta["benchmark_architecture"]["n"]   (Task 3 section 1)
      4) parse from meta["tag"] like "..._l3_n16_..."                               (fallback)
    """
    if "num_hidden_layers" in meta and "nodes_per_layer" in meta:
        return int(meta["num_hidden_layers"]), int(meta["nodes_per_layer"])

    arch = meta.get("architecture", {}) or {}
    if "l" in arch and "n" in arch:
        return int(arch["l"]), int(arch["n"])

    bench = meta.get("benchmark_architecture", {}) or {}
    if "l" in bench and "n" in bench:
        return int(bench["l"]), int(bench["n"])

    tag = str(meta.get("tag", ""))
    m = re.search(r"_l(\d+)_n(\d+)_", tag)
    if m:
        return int(m.group(1)), int(m.group(2))

    raise KeyError(
        "Could not find architecture in meta (expected num_hidden_layers/nodes_per_layer "
        "or architecture.{l,n} or benchmark_architecture.{l,n}, or parseable _l*_n* in tag)."
    )


def _build_like_model_from_meta(
    meta: dict[str, Any],
    *,
    like_key: jrandom.PRNGKey,
    dataset_1: dict | None,
    G_cub,
):
    """
    Constructs the correct model *structure* required by eqx.tree_deserialise_leaves.
    """
    mid = str(meta.get("model_id", "")).upper().strip()
    mtype = str(meta.get("model_type", "")).upper().strip()

    # Architecture
    l, n = _get_arch_from_meta(meta)

    # Activation: your workflows always save "softplus" in meta; keep it fixed here.
    activation = jax.nn.softplus

    # ------------------------------------------------------------
    # Task 2 / MS, MSW (stress model: C->6 -> P(9))
    # ------------------------------------------------------------
    if mid in ("MS", "MSW"):
        return tm.build(
            key=like_key,
            input_dim=6,
            output_dim=9,
            num_hidden_layers=l,
            nodes_per_layer=n,
            activations=activation,
            constrain_icnn_weights=False,
        )

    # ------------------------------------------------------------
    # Task 3 / WITI (energy+stress from invariants; needs G_ti)
    # ------------------------------------------------------------
    if mid == "WITI":
        if dataset_1 is None:
            raise ValueError("Loading WITI requires dataset_1=prepare_dataset_1(...) so G_ti is available.")
        G_ti = dataset_1["G_ti"]
        return tm.SobolevModel_WI_ti(
            G_ti=G_ti,
            key=like_key,
            input_dim=5,
            output_dim="scalar",
            num_hidden_layers=l,
            nodes_per_layer=n,
            activation=activation,
            is_icnn=False,
            is_ficnn=True,
        )

    # ------------------------------------------------------------
    # Task 5.2 / WICUB (cubic invariants; needs G_cub)
    # ------------------------------------------------------------
    if mid == "WICUB":
        if G_cub is None:
            raise ValueError("Loading WICUB requires G_cub=... (same tensor used in prepare_dataset_3 / training).")
        return tm.SobolevModel_WI_Cubic(
            G_cub=G_cub,
            key=like_key,
            input_dim=6,
            output_dim="scalar",
            num_hidden_layers=l,
            nodes_per_layer=n,
            activation=activation,
            is_icnn=False,
            is_ficnn=True,
        )

    # ------------------------------------------------------------
    # Task 5.3 / WF and Task 5.4 / WF_AUG (polyconvex ICNN W(F))
    # ------------------------------------------------------------
    if mid in ("WF", "WF_AUG"):
        icnn = meta.get("icnn", {}) or {}
        is_icnn = bool(icnn.get("is_icnn", True))
        is_ficnn = bool(icnn.get("is_ficnn", False))
        return tm.SobolevModel_WF(
            key=like_key,
            input_dim=19,
            output_dim="scalar",
            num_hidden_layers=l,
            nodes_per_layer=n,
            activation=activation,
            is_icnn=is_icnn,
            is_ficnn=is_ficnn,
        )

    # ------------------------------------------------------------
    # Task 4 (meta["model_type"] = "FFNN" or "PANN")
    # Note: your repo uses SobolevModel_WI_ti; some cells call SobolevModel_WI.
    # We support both by falling back to SobolevModel_WI_ti.
    # ------------------------------------------------------------
    if mtype in ("FFNN", "PANN"):
        if mtype == "FFNN":
            return tm.build(
                key=like_key,
                input_dim=6,
                output_dim=9,
                num_hidden_layers=l,
                nodes_per_layer=n,
                activations=activation,
                constrain_icnn_weights=False,
            )

        # PANN (WI TI)
        if dataset_1 is None:
            raise ValueError("Loading Task-4 PANN requires dataset_1 or at least a dict containing G_ti.")
        G_ti = dataset_1["G_ti"]

        WI_cls = getattr(tm, "SobolevModel_WI", None)
        if WI_cls is None:
            WI_cls = tm.SobolevModel_WI_ti

        return WI_cls(
            G_ti=G_ti,
            key=like_key,
            input_dim=5,
            output_dim="scalar",
            num_hidden_layers=l,
            nodes_per_layer=n,
            activation=activation,
            is_icnn=False,
            is_ficnn=True,
        )

    raise ValueError(
        f"Unknown model in meta. model_id='{mid}', model_type='{mtype}'. "
        "Extend _build_like_model_from_meta for this case."
    )

#---------- Parallelized

def _run_single_task2_3_msw(args):
    """
    Single MSW weighted training run for Task 2.3 (one init), with persistence.

    Notes:
      - Must be top-level for joblib (Windows spawn).
      - Avoids relying on outer scope.
    """
    import json
    import pickle
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import klax

    from . import models as tm
    from . import losses as tl

    (
        init_idx,                 # 1-based init index
        X_cal_MS, Y_cal_MS, sample_weights,
        steps, batch_size, learning_rate,
        base_seed,
        w_uni, w_ps, w_bi, w_uni_inv, w_ps_inv, w_bi_inv,
        out_dir,
    ) = args

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic keys per init
    model_key = jrandom.PRNGKey(base_seed + init_idx * 1000 + 1)
    train_key = jrandom.PRNGKey(base_seed + init_idx * 1000 + 2)

    # Weighted training data format: (X, (Y, weights))
    train_data_MS_w = (X_cal_MS, (Y_cal_MS, sample_weights))

    # Loss function
    loss_fn_MS_w = tl.WeightedMSE()

    try:
        # MEDIUM architecture: l=3, n=16
        model = tm.build(
            key=model_key,
            input_dim=6,
            output_dim=9,
            num_hidden_layers=3,
            nodes_per_layer=16,
            activations=jax.nn.softplus,
            constrain_icnn_weights=False
        )

        trained, history = tm.train_model(
            model,
            train_data_MS_w,
            train_key,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_fn=loss_fn_MS_w
        )

        final_model = klax.finalize(trained)

        # Save artifacts
        tag = f"MSW_medium_l3_n16_steps{steps}_init{init_idx:02d}"
        model_path   = out_dir / f"{tag}.eqx"
        history_path = out_dir / f"{tag}_history.pkl"
        meta_path    = out_dir / f"{tag}.json"

        # Persist
        import equinox as eqx
        eqx.tree_serialise_leaves(str(model_path), final_model)
        with open(history_path, "wb") as f:
            pickle.dump(history, f)

        meta = {
            "task": "2.3",
            "model_id": "MSW",
            "tag": tag,
            "architecture_note": "MEDIUM baseline from Task 2.2: l=3, n=16, softplus",
            "comparison_note": "Compare to Task 2.2 MS (unweighted) with l=3, n=16 and steps=300000",
            "num_hidden_layers": 3,
            "nodes_per_layer": 16,
            "activation": "softplus",
            "steps": int(steps),
            "init_idx": int(init_idx),
            "n_inits": None,  # filled by caller if desired; kept optional here
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "loss": "WeightedMSE",
            "weights": {
                "w_uni": float(w_uni),
                "w_ps": float(w_ps),
                "w_bi": float(w_bi),
                "w_uni_inv": float(w_uni_inv),
                "w_ps_inv": float(w_ps_inv),
                "w_bi_inv": float(w_bi_inv),
            },
            "base_seed": int(base_seed),
            "saved_model_path": str(model_path),
            "saved_history_path": str(history_path),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta

    except Exception as e:
        # Return something useful; also prints which init failed
        print(f"[Task2.3][MSW] init{init_idx:02d} failed: {e}")
        return {
            "task": "2.3",
            "model_id": "MSW",
            "init_idx": int(init_idx),
            "steps": int(steps),
            "error": str(e),
        }

def workflow_task_2_3_train_ms_weighted_5inits(
    dataset_1: dict,
    *,
    out_dir: str | Path = "artifacts/task2_3",
    n_inits: int = 5,
    steps: int = 300_000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    n_jobs: int = -2,
    backend: str = "loky",
    max_nbytes: str | None = "50M",
    verbose: int | None = None,
    base_seed: int | None = None,
):
    """
    Task 2.3: Loss-weighted MS model (MSW) training with multiple random initializations.

    Parallelization:
      - uses joblib over init dimension
      - each init writes its own artifacts, so no write conflicts

    Parameters added:
      - n_jobs, backend, max_nbytes, verbose
      - base_seed (optional): to make init keys stable across runs/machines
    """
    from joblib import Parallel, delayed

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
    # IMPORTANT: order must match concatenation above: [uni, ps, bi]
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

    # ------------------------------------------------------------
    # Stable base_seed (so init keys are deterministic)
    # ------------------------------------------------------------
    if base_seed is None:
        # deterministic default from master_key (but converted to python int)
        mk = dataset_1["master_key"]
        base_seed = int(jrandom.randint(mk, (), 0, 10_000_000))

    # ------------------------------------------------------------
    # Build experiments (one per init)
    # ------------------------------------------------------------
    experiments = []
    for init_idx in range(1, n_inits + 1):
        experiments.append((
            init_idx,
            X_cal_MS, Y_cal_MS, sample_weights,
            steps, batch_size, learning_rate,
            base_seed,
            w_uni, w_ps, w_bi, w_uni_inv, w_ps_inv, w_bi_inv,
            str(out_dir),
        ))

    total = len(experiments)
    if verbose is None:
        # similar spirit to your Task 4: show progress
        verbose = total

    print(f"Task 2.3 (MSW): training {n_inits} initializations in parallel (n_jobs={n_jobs}, backend={backend})")

    results = Parallel(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        max_nbytes=max_nbytes,
    )(
        delayed(_run_single_task2_3_msw)(exp) for exp in experiments
    )

    # Optional: fill n_inits in each successful meta
    for m in results:
        if isinstance(m, dict) and m.get("model_id") == "MSW" and "error" not in m:
            m["n_inits"] = int(n_inits)

    return results

def _run_single_task3_witi(args):
    """
    Single training run for Task 3: (strategy, init) of WI_ti benchmark model.

    Must be top-level for joblib on Windows (loky backend).
    """
    import json
    import pickle
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import equinox as eqx
    import klax

    from . import models as tm
    from . import losses as tl
    from . import data_t2 as td2

    (
        strat_name, init_idx,                      # e.g. "A", 1..n
        G_ti,
        F_cal_all, I_cal_all, W_cal_all, P_cal_all,
        sample_weights,
        steps, batch_size, learning_rate,
        alpha, beta,
        base_seed,
        w_uni, w_ps, w_bi, w_uni_inv, w_ps_inv, w_bi_inv,
        out_dir,
    ) = args

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # deterministic keys per (strategy, init)
    # offset strat so A/B/C never collide
    strat_offset = {"A": 100_000, "B": 200_000, "C": 300_000}.get(strat_name, 999_000)
    seed0 = base_seed + strat_offset + init_idx * 1000

    model_key = jrandom.PRNGKey(seed0 + 1)
    train_key = jrandom.PRNGKey(seed0 + 2)

    # Train data format for WeightedSobolevLoss:
    # batch = (x, ((W_true, P_true), w))
    train_data = (
        (F_cal_all, I_cal_all),
        ((W_cal_all, P_cal_all), sample_weights),
    )

    loss_fn = tl.WeightedSobolevLoss(alpha=float(alpha), beta=float(beta))

    tag = f"WITI_{strat_name}_bench_l3_n16_steps{steps}_init{init_idx:02d}"
    model_path   = out_dir / f"{tag}.eqx"
    history_path = out_dir / f"{tag}_history.pkl"
    meta_path    = out_dir / f"{tag}.json"

    try:
        # Benchmark WI_ti model (l=3, n=16, softplus, FICNN enabled)
        model = tm.SobolevModel_WI_ti(
            G_ti=G_ti,
            key=model_key,
            input_dim=5,
            output_dim="scalar",
            num_hidden_layers=3,
            nodes_per_layer=16,
            activation=jax.nn.softplus,
            is_icnn=False,
            is_ficnn=True,
        )

        trained_model, history = tm.train_WI(
            model=model,
            train_data=train_data,
            key=train_key,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
        )

        final_model = klax.finalize(trained_model)

        # Persist
        eqx.tree_serialise_leaves(str(model_path), final_model)
        with open(history_path, "wb") as f:
            pickle.dump(history, f)

        meta = {
            "task": "3",
            "model_id": "WITI",
            "strategy": strat_name,
            "tag": tag,
            "benchmark_architecture": {"l": 3, "n": 16, "activation": "softplus"},
            "steps": int(steps),
            "init_idx": int(init_idx),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "loss": "WeightedSobolevLoss",
            "loss_params": {"alpha": float(alpha), "beta": float(beta)},
            "loss_weighting": "inverse path weights (uniaxial, pure shear, biaxial)",
            "path_weights": {
                "w_uni": float(w_uni), "w_ps": float(w_ps), "w_bi": float(w_bi),
                "w_uni_inv": float(w_uni_inv), "w_ps_inv": float(w_ps_inv), "w_bi_inv": float(w_bi_inv),
                "concat_order": "[biaxial, uniaxial, pure_shear]",
            },
            "saved_model_path": str(model_path),
            "saved_history_path": str(history_path),
            "base_seed": int(base_seed),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta

    except Exception as e:
        print(f"[Task3][WITI] strategy={strat_name} init={init_idx:02d} FAILED: {e}")
        return {
            "task": "3",
            "model_id": "WITI",
            "strategy": strat_name,
            "init_idx": int(init_idx),
            "steps": int(steps),
            "error": str(e),
        }

def workflow_task_3_train_wi_ti_strategies_abc(
    dataset_1: dict,
    *,
    out_dir: str | Path = "artifacts/task3",
    n_inits: int = 5,
    steps: int = 300_000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    n_jobs: int = -2,
    backend: str = "loky",
    max_nbytes: str | None = "50M",
    verbose: int | None = None,
    base_seed: int = 42,
):
    """
    Task 3 workflow: Train WI_ti benchmark model under strategies A/B/C
    with n_inits random initializations each, in parallel using joblib.
    """
    from joblib import Parallel, delayed

    out_dir = _ensure_dir(out_dir)

    G_ti = dataset_1["G_ti"]

    # ------------------------------------------------------------
    # Build calibration dataset with the order you used in Task 3:
    # IMPORTANT: order [biaxial, uniaxial, pure_shear]
    # ------------------------------------------------------------
    F_bi, F_uni, F_ps = dataset_1["F_bi"], dataset_1["F_uni"], dataset_1["F_ps"]
    P_bi, P_uni, P_ps = dataset_1["P_bi"], dataset_1["P_uni"], dataset_1["P_ps"]
    W_bi, W_uni, W_ps = dataset_1["W_bi"], dataset_1["W_uni"], dataset_1["W_ps"]
    I_bi, I_uni, I_ps = dataset_1["I_bi"], dataset_1["I_uni"], dataset_1["I_ps"]

    F_cal_all = jnp.concatenate([F_bi, F_uni, F_ps], axis=0)
    I_cal_all = jnp.concatenate([I_bi, I_uni, I_ps], axis=0)
    W_cal_all = jnp.concatenate([W_bi, W_uni, W_ps], axis=0)
    P_cal_all = jnp.concatenate([P_bi, P_uni, P_ps], axis=0)

    # ------------------------------------------------------------
    # Compute inverse path weights
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

    # IMPORTANT: must match F_cal_all concat order [bi, uni, ps]
    sample_weights = jnp.concatenate([weights_bi, weights_uni, weights_ps], axis=0).reshape(-1)

    # ------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------
    strategies = {
        "A": (1.0, 0.0),
        "B": (0.0, 1.0),
        "C": (1.0, 1.0),
    }

    # ------------------------------------------------------------
    # Build experiments list (strategy × init)
    # ------------------------------------------------------------
    experiments = []
    for strat_name, (alpha, beta) in strategies.items():
        for init_idx in range(1, n_inits + 1):
            experiments.append((
                strat_name, init_idx,
                G_ti,
                F_cal_all, I_cal_all, W_cal_all, P_cal_all,
                sample_weights,
                steps, batch_size, learning_rate,
                alpha, beta,
                base_seed,
                w_uni, w_ps, w_bi, w_uni_inv, w_ps_inv, w_bi_inv,
                str(out_dir),
            ))

    total = len(experiments)
    if verbose is None:
        verbose = total

    print(f"Task 3: 3 strategies × {n_inits} inits = {total} trainings (n_jobs={n_jobs}, backend={backend})")

    results = Parallel(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        max_nbytes=max_nbytes,
    )(
        delayed(_run_single_task3_witi)(exp) for exp in experiments
    )

    return results

def _run_single_task3_witi_arch_steps(args):
    """
    Single run for Task 3 – Part 2:
    (l, n, steps, init) for fixed strategy.
    """
    import json
    import pickle
    from pathlib import Path

    import jax
    import jax.random as jrandom
    import equinox as eqx
    import klax

    from . import models as tm
    from . import losses as tl

    (
        l, n, steps, init_idx,
        strategy, alpha, beta,
        G_ti,
        F_cal_all, I_cal_all, W_cal_all, P_cal_all,
        sample_weights,
        batch_size, learning_rate,
        base_seed,
        out_dir,
    ) = args

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed0 = base_seed + 10_000 * l + 1_000 * n + 10 * init_idx
    model_key = jrandom.PRNGKey(seed0 + 1)
    train_key = jrandom.PRNGKey(seed0 + 2)

    train_data = (
        (F_cal_all, I_cal_all),
        ((W_cal_all, P_cal_all), sample_weights),
    )

    loss_fn = tl.WeightedSobolevLoss(alpha=alpha, beta=beta)

    tag = f"WITI_{strategy}_l{l}_n{n}_steps{steps}_init{init_idx:02d}"
    model_path = out_dir / f"{tag}.eqx"
    history_path = out_dir / f"{tag}_history.pkl"
    meta_path = out_dir / f"{tag}.json"

    try:
        model = tm.SobolevModel_WI_ti(
            G_ti=G_ti,
            key=model_key,
            input_dim=5,
            output_dim="scalar",
            num_hidden_layers=l,
            nodes_per_layer=n,
            activation=jax.nn.softplus,
            is_icnn=False,
            is_ficnn=True,
        )

        trained, history = tm.train_WI(
            model=model,
            train_data=train_data,
            key=train_key,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
        )

        final_model = klax.finalize(trained)

        eqx.tree_serialise_leaves(str(model_path), final_model)
        with open(history_path, "wb") as f:
            pickle.dump(history, f)

        meta = {
            "task": "3.2",
            "model_id": "WITI",
            "strategy": strategy,
            "tag": tag,
            "architecture": {"l": l, "n": n, "activation": "softplus"},
            "steps": steps,
            "init_idx": init_idx,
            "loss": "WeightedSobolevLoss",
            "loss_params": {"alpha": alpha, "beta": beta},
            "saved_model_path": str(model_path),
            "saved_history_path": str(history_path),
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return meta

    except Exception as e:
        print(f"[Task3.2] FAIL l={l} n={n} steps={steps} init={init_idx}: {e}")
        return {"error": str(e)}

def workflow_task_3_sweep_wi_ti_arch_steps(
    dataset_1,
    *,
    strategy: str,
    out_dir="artifacts/task3_section2",
    n_inits=5,
    steps_list=(100_000, 300_000, 500_000),
    archs=((2, 8), (3, 16), (4, 32)),
    batch_size=32,
    learning_rate=1e-3,
    n_jobs=-2,
    backend="loky",
    base_seed=42,
):
    from joblib import Parallel, delayed
    from . import data_t2 as td2

    out_dir = _ensure_dir(out_dir)

    # Calibration data (same as Part 1)
    G_ti = dataset_1["G_ti"]

    F_cal_all = jnp.concatenate(
        [dataset_1["F_bi"], dataset_1["F_uni"], dataset_1["F_ps"]], axis=0
    )
    I_cal_all = jnp.concatenate(
        [dataset_1["I_bi"], dataset_1["I_uni"], dataset_1["I_ps"]], axis=0
    )
    W_cal_all = jnp.concatenate(
        [dataset_1["W_bi"], dataset_1["W_uni"], dataset_1["W_ps"]], axis=0
    )
    P_cal_all = jnp.concatenate(
        [dataset_1["P_bi"], dataset_1["P_uni"], dataset_1["P_ps"]], axis=0
    )

    w_uni = td2.compute_path_weight(dataset_1["P_uni"])
    w_ps  = td2.compute_path_weight(dataset_1["P_ps"])
    w_bi  = td2.compute_path_weight(dataset_1["P_bi"])

    weights = jnp.concatenate([
        (1.0 / w_bi)  * jnp.ones(dataset_1["P_bi"].shape[0]),
        (1.0 / w_uni) * jnp.ones(dataset_1["P_uni"].shape[0]),
        (1.0 / w_ps)  * jnp.ones(dataset_1["P_ps"].shape[0]),
    ])

    # Strategy parameters
    alpha, beta = {"A": (1, 0), "B": (0, 1), "C": (1, 1)}[strategy]

    experiments = []
    for (l, n) in archs:
        for steps in steps_list:
            for init_idx in range(1, n_inits + 1):
                experiments.append((
                    l, n, steps, init_idx,
                    strategy, alpha, beta,
                    G_ti,
                    F_cal_all, I_cal_all, W_cal_all, P_cal_all,
                    weights,
                    batch_size, learning_rate,
                    base_seed,
                    str(out_dir),
                ))

    return Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_run_single_task3_witi_arch_steps)(e) for e in experiments
    )

def _run_single_task3_section2(exp):
    """
    One training run for Task 3 - Section 2:
      strategy + (arch_name,l,n) + steps + init_idx

    exp is a tuple with all data needed so it is joblib-picklable.
    """
    import json
    import pickle
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import equinox as eqx
    import klax

    from . import models as tm
    from . import losses as tl

    (
        strategy, arch_name, l, n, steps, init_idx,
        # calibration data
        G_ti,
        F_cal_all, I_cal_all, W_cal_all, P_cal_all,
        sample_weights,
        # hyperparams
        batch_size, learning_rate,
        # deterministic seeding
        base_seed,
        # output
        out_dir,
    ) = exp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # deterministic-ish run seed
    # include all config knobs so parallel scheduling does not change results
    seed = (
        int(base_seed)
        + 100_000 * (ord(strategy) - ord("A") + 1)
        + 10_000 * (1 if arch_name == "small" else 2 if arch_name == "medium" else 3)
        + 1_000 * int(steps // 100_000)
        + 10 * int(l)
        + int(n)
        + int(init_idx)
    )
    model_key = jrandom.PRNGKey(seed + 1)
    train_key = jrandom.PRNGKey(seed + 2)

    # loss strategy
    strategy = strategy.upper()
    if strategy == "A":
        loss_fn = tl.WeightedSobolevLoss(alpha=1.0, beta=0.0)
    elif strategy == "B":
        loss_fn = tl.WeightedSobolevLoss(alpha=0.0, beta=1.0)
    elif strategy == "C":
        loss_fn = tl.WeightedSobolevLoss(alpha=1.0, beta=1.0)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    train_data = (
        (F_cal_all, I_cal_all),
        ((W_cal_all, P_cal_all), sample_weights),
    )

    tag = f"WITI_{strategy}_{arch_name}_l{l}_n{n}_steps{steps}_init{init_idx:02d}"
    model_path   = out_dir / f"{tag}.eqx"
    history_path = out_dir / f"{tag}_history.pkl"
    meta_path    = out_dir / f"{tag}.json"

    try:
        model = tm.SobolevModel_WI_ti(
            G_ti=G_ti,
            key=model_key,
            input_dim=5,
            output_dim="scalar",
            num_hidden_layers=l,
            nodes_per_layer=n,
            activation=jax.nn.softplus,
            is_icnn=False,
            is_ficnn=True,
        )

        trained_model, history = tm.train_WI(
            model=model,
            train_data=train_data,
            key=train_key,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
        )

        final_model = klax.finalize(trained_model)

        eqx.tree_serialise_leaves(str(model_path), final_model)
        with open(history_path, "wb") as f:
            pickle.dump(history, f)

        meta = {
            "task": "3_section2",
            "model_id": "WITI",
            "strategy": strategy,
            "tag": tag,
            "num_hidden_layers": int(l),
            "nodes_per_layer": int(n),
            "activation": "softplus",
            "steps": int(steps),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "loss": "WeightedSobolevLoss",
            "loss_params": {"alpha": float(loss_fn.alpha), "beta": float(loss_fn.beta)},
            "loss_weighting": "inverse path weights (uniaxial, pure shear, biaxial)",
            "saved_model_path": str(model_path),
            "saved_history_path": str(history_path),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta

    except Exception as e:
        # fail-safe: do not crash the whole Parallel call
        print(f"[Task3-Section2] FAIL: {tag} -> {e}")
        return {"error": str(e), "tag": tag}

def workflow_task_3_sweep_wi_ti_arch_steps(
    dataset_1: dict,
    *,
    strategy: str = "C",
    out_dir: str | Path = "artifacts/task3_section2",
    n_inits: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    archs=None,
    steps_list=None,
    # joblib controls
    n_jobs: int = -2,
    backend: str = "loky",
    verbose: int = 10,
    base_seed: int | None = None,
):
    """
    Parallel version of Task 3 (Section 2): arch × steps × init.
    Uses the same variants as your serial workflow.
    """
    from pathlib import Path
    from joblib import Parallel, delayed
    import jax.numpy as jnp
    import jax.random as jrandom

    from . import data_t2 as td2  # for compute_path_weight

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy = strategy.upper()
    assert strategy in ("A", "B", "C"), "strategy must be 'A', 'B', or 'C'"

    if archs is None:
        archs = [
            ("small",  2,  8),
            ("medium", 3, 16),
            ("large",  4, 32),
        ]

    if steps_list is None:
        steps_list = [100_000, 300_000, 500_000]

    G_ti = dataset_1["G_ti"]
    master_key = dataset_1["master_key"]

    # stable base seed (so parallelism doesn’t change results)
    if base_seed is None:
        base_seed = int(jrandom.randint(master_key, (), 0, 10_000_000))

    # ------------------------------------------------------------
    # Calibration data (same ordering as Task 3 Section 1):
    # IMPORTANT: order [biaxial, uniaxial, pure_shear]
    # ------------------------------------------------------------
    F_bi  = dataset_1["F_bi"]
    F_uni = dataset_1["F_uni"]
    F_ps  = dataset_1["F_ps"]

    P_bi  = dataset_1["P_bi"]
    P_uni = dataset_1["P_uni"]
    P_ps  = dataset_1["P_ps"]

    W_bi  = dataset_1["W_bi"]
    W_uni = dataset_1["W_uni"]
    W_ps  = dataset_1["W_ps"]

    I_bi  = dataset_1["I_bi"]
    I_uni = dataset_1["I_uni"]
    I_ps  = dataset_1["I_ps"]

    F_cal_all = jnp.concatenate([F_bi, F_uni, F_ps], axis=0)
    I_cal_all = jnp.concatenate([I_bi, I_uni, I_ps], axis=0)
    W_cal_all = jnp.concatenate([W_bi, W_uni, W_ps], axis=0)
    P_cal_all = jnp.concatenate([P_bi, P_uni, P_ps], axis=0)

    # ------------------------------------------------------------
    # Loss-weighting: inverse path weights
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

    sample_weights = jnp.concatenate([weights_bi, weights_uni, weights_ps], axis=0).reshape(-1)

    # ------------------------------------------------------------
    # Build experiments list
    # ------------------------------------------------------------
    experiments = []
    for arch_name, l, n in archs:
        for steps in steps_list:
            for init_idx in range(1, n_inits + 1):
                experiments.append((
                    strategy, arch_name, l, n, int(steps), int(init_idx),
                    G_ti,
                    F_cal_all, I_cal_all, W_cal_all, P_cal_all,
                    sample_weights,
                    int(batch_size), float(learning_rate),
                    int(base_seed),
                    str(out_dir),
                ))

    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(_run_single_task3_section2)(exp) for exp in experiments
    )

    return results

def _run_single_task3_calibration_study(exp):
    """
    One training run for Task 3 (calibration subset study):
      subset (tuple of path names) + init

    exp is joblib-picklable (no closures).
    """
    import json
    import pickle
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import equinox as eqx
    import klax

    from . import models as tm
    from . import data_t2 as td2
    from . import losses as tl

    (
        subset, subset_tag,
        best_l, best_n, best_strategy, steps,
        # calibration paths data:
        cal_paths,
        G_ti,
        batch_size, learning_rate,
        init_idx,
        base_seed,
        out_dir,
    ) = exp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # loss strategy
    best_strategy = best_strategy.upper()
    if best_strategy == "A":
        loss_fn = tl.WeightedSobolevLoss(alpha=1.0, beta=0.0)
    elif best_strategy == "B":
        loss_fn = tl.WeightedSobolevLoss(alpha=0.0, beta=1.0)
    elif best_strategy == "C":
        loss_fn = tl.WeightedSobolevLoss(alpha=1.0, beta=1.0)
    else:
        raise ValueError(f"Unknown strategy '{best_strategy}'")

    # deterministic-ish seed
    subset_hash = sum((i + 1) * (len(s) + ord(s[0])) for i, s in enumerate(subset))
    seed = int(base_seed) + 1000 * subset_hash + 10 * int(init_idx)
    model_key = jrandom.PRNGKey(seed + 1)
    train_key = jrandom.PRNGKey(seed + 2)

    # Build train_data for subset (exactly like your serial helper)
    F_all = jnp.concatenate([cal_paths[p]["F"] for p in subset], axis=0)
    I_all = jnp.concatenate([cal_paths[p]["I"] for p in subset], axis=0)
    W_all = jnp.concatenate([cal_paths[p]["W"] for p in subset], axis=0)
    P_all = jnp.concatenate([cal_paths[p]["P"] for p in subset], axis=0)

    weights_list = []
    weights_meta = {}
    for p in subset:
        P_path = cal_paths[p]["P"]
        w_path = td2.compute_path_weight(P_path)
        w_inv = 1.0 / w_path
        weights_list.append(w_inv * jnp.ones(P_path.shape[0]))
        weights_meta[p] = {"w": float(w_path), "w_inv": float(w_inv), "n_samples": int(P_path.shape[0])}

    sample_weights = jnp.concatenate(weights_list, axis=0).reshape(-1)

    train_data = (
        (F_all, I_all),
        ((W_all, P_all), sample_weights),
    )

    tag = (
        f"WITI_CAL_{subset_tag}_{best_strategy}"
        f"_l{best_l}_n{best_n}_steps{steps}"
        f"_init{init_idx:02d}"
    )

    model_path   = out_dir / f"{tag}.eqx"
    history_path = out_dir / f"{tag}_history.pkl"
    meta_path    = out_dir / f"{tag}.json"

    try:
        model = tm.SobolevModel_WI_ti(
            G_ti=G_ti,
            key=model_key,
            input_dim=5,
            output_dim="scalar",
            num_hidden_layers=int(best_l),
            nodes_per_layer=int(best_n),
            activation=jax.nn.softplus,
            is_icnn=False,
            is_ficnn=True,
        )

        trained_model, history = tm.train_WI(
            model=model,
            train_data=train_data,
            key=train_key,
            steps=int(steps),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            loss_fn=loss_fn,
        )

        final_model = klax.finalize(trained_model)

        eqx.tree_serialise_leaves(str(model_path), final_model)
        with open(history_path, "wb") as f:
            pickle.dump(history, f)

        meta = {
            "task": "3_calibration_study",
            "model_id": "WITI",
            "tag": tag,
            "subset": list(subset),
            "subset_tag": subset_tag,
            "architecture": {"l": int(best_l), "n": int(best_n), "activation": "softplus"},
            "strategy": best_strategy,
            "loss": "WeightedSobolevLoss",
            "loss_params": {"alpha": float(loss_fn.alpha), "beta": float(loss_fn.beta)},
            "steps": int(steps),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "loss_weighting": "inverse path weights computed on subset paths",
            "weights_by_path": weights_meta,
            "saved_model_path": str(model_path),
            "saved_history_path": str(history_path),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta

    except Exception as e:
        print(f"[Task3-CalibStudy] FAIL: {tag} -> {e}")
        return {"error": str(e), "tag": tag}

def workflow_task_3_calibration_set_study(
    dataset_1: dict,
    *,
    best_l: int = 3,
    best_n: int = 16,
    best_strategy: str = "C",
    steps: int = 300_000,
    out_dir: str | Path = "artifacts/task3_calibration_study",
    n_inits: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    include_single_paths: bool = True,
    include_pair_paths: bool = True,
    include_all_paths: bool = True,
    # joblib controls
    n_jobs: int = -2,
    backend: str = "loky",
    verbose: int = 10,
    base_seed: int | None = None,
):
    """
    Parallel version of Task 3 (final section): calibration subset study.
    Uses the exact same subsets + naming as your serial workflow.
    """
    from pathlib import Path
    import itertools
    from joblib import Parallel, delayed
    import jax.random as jrandom

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_strategy = best_strategy.upper()
    assert best_strategy in ("A", "B", "C"), "best_strategy must be 'A', 'B', or 'C'"

    G_ti = dataset_1["G_ti"]
    master_key = dataset_1["master_key"]

    if base_seed is None:
        base_seed = int(jrandom.randint(master_key, (), 0, 10_000_000))

    # Available calibration paths (exactly like your serial workflow)
    cal_paths = {
        "biaxial": {
            "F": dataset_1["F_bi"],
            "I": dataset_1["I_bi"],
            "W": dataset_1["W_bi"],
            "P": dataset_1["P_bi"],
        },
        "uniaxial": {
            "F": dataset_1["F_uni"],
            "I": dataset_1["I_uni"],
            "W": dataset_1["W_uni"],
            "P": dataset_1["P_uni"],
        },
        "pure_shear": {
            "F": dataset_1["F_ps"],
            "I": dataset_1["I_ps"],
            "W": dataset_1["W_ps"],
            "P": dataset_1["P_ps"],
        },
    }

    path_names = list(cal_paths.keys())

    # Build subsets (exactly like before)
    subsets = []
    if include_single_paths:
        subsets += [(p,) for p in path_names]
    if include_pair_paths:
        subsets += list(itertools.combinations(path_names, 2))
    if include_all_paths:
        subsets += [tuple(path_names)]

    # Build experiments list: subset × init
    experiments = []
    for subset in subsets:
        subset_tag = "+".join(subset)  # exact naming you used before
        for init_idx in range(1, n_inits + 1):
            experiments.append((
                tuple(subset), subset_tag,
                int(best_l), int(best_n), best_strategy, int(steps),
                cal_paths,
                G_ti,
                int(batch_size), float(learning_rate),
                int(init_idx),
                int(base_seed),
                str(out_dir),
            ))

    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(_run_single_task3_calibration_study)(exp) for exp in experiments
    )

    return results

def _run_single_task5_2_wicub(exp):
    """
    One training run for Task 5.2 (WI_cubic):
      (arch, steps, init) with persistence.

    Must be top-level for joblib (Windows loky).
    """
    import json
    from pathlib import Path

    import jax
    import equinox as eqx
    import klax

    from . import models as tm
    from . import losses as tl

    (
        arch_name, l, n, steps, init_idx,
        model_key, train_key,
        G_cub,
        train_data,
        batch_size, learning_rate,
        loss_alpha, loss_beta,
        strategy_tag,
        out_dir,
        dataset3_meta,
    ) = exp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        loss_fn = tl.WeightedSobolevLoss(alpha=loss_alpha, beta=loss_beta)

        model = tm.SobolevModel_WI_Cubic(
            G_cub=G_cub,
            key=model_key,
            input_dim=6,
            output_dim="scalar",
            num_hidden_layers=l,
            nodes_per_layer=n,
            activation=jax.nn.softplus,
            is_icnn=False,
            is_ficnn=True,
        )

        trained_model, history = tm.train_model(
            model,
            train_data,
            train_key,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
        )

        final_model = klax.finalize(trained_model)

        tag = f"WICUB_{strategy_tag}_{arch_name}_l{l}_n{n}_steps{steps}_init{init_idx:02d}"
        model_path = out_dir / f"{tag}.eqx"
        hist_path  = out_dir / f"{tag}_history.pkl"
        meta_path  = out_dir / f"{tag}.json"

        # Use your project helpers if available; otherwise serialize directly.
        # (Your serial workflow uses _save_eqx_model/_save_history.)
        try:
            _save_eqx_model(final_model, model_path)   # noqa: F821
            _save_history(history, hist_path)          # noqa: F821
        except Exception:
            eqx.tree_serialise_leaves(str(model_path), final_model)
            import pickle
            with open(hist_path, "wb") as f:
                pickle.dump(history, f)

        meta = {
            "task": "5.2",
            "model_id": "WICUB",
            "tag": tag,
            "architecture": {"l": l, "n": n, "activation": "softplus"},
            "steps": int(steps),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "loss": "WeightedSobolevLoss",
            "loss_params": {"alpha": float(loss_alpha), "beta": float(loss_beta)},
            "dataset_3": dataset3_meta,
            "saved_model_path": str(model_path),
            "saved_history_path": str(hist_path),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta

    except Exception as e:
        print(f"[Task5.2][WICUB] FAIL arch={arch_name} steps={steps} init={init_idx:02d}: {e}")
        return {
            "task": "5.2",
            "model_id": "WICUB",
            "arch": arch_name,
            "l": int(l),
            "n": int(n),
            "steps": int(steps),
            "init_idx": int(init_idx),
            "error": str(e),
        }

def workflow_task_5_2_sweep_wi_cubic(
    dataset_3: dict,
    *,
    G_cub: jnp.ndarray,
    out_dir: str | Path = "artifacts/task5_2",
    n_inits: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    steps_list=None,
    archs=None,
    loss_alpha: float = 1.0,
    loss_beta: float = 0.0,
    master_key: jrandom.PRNGKey = jrandom.PRNGKey(0),
    # ---- joblib knobs ----
    n_jobs: int = -2,
    backend: str = "loky",
    verbose: int | None = None,
    max_nbytes: str | None = "50M",
):
    """
    Task 5.2: Parallel sweep WI_cubic model configurations on Dataset 3.

    Parallel unit = (arch, steps, init).
    """
    from joblib import Parallel, delayed
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if steps_list is None:
        steps_list = [100_000, 300_000, 500_000]

    if archs is None:
        archs = [
            ("small",  2,  8),
            ("medium", 3, 16),
            ("large",  4, 32),
        ]

    train_data = dataset_3["train_data_WI_cubic"]
    # test_data exists but isn’t needed for training persistence
    # test_data = dataset_3["test_data_WI_cubic"]

    strategy_tag = f"a{loss_alpha:g}_b{loss_beta:g}"

    # Preserve deterministic key assignment like your serial workflow
    total_runs = len(archs) * len(steps_list) * n_inits
    keys = jrandom.split(master_key, total_runs * 2 + 1)
    key_cursor = 1

    # Small meta payload (pickle-friendly)
    dataset3_meta = {
        "path_h5": dataset_3.get("path_h5", None),
        "alpha_scale": float(dataset_3["alpha"]),
        "P_max": float(dataset_3["P_max"]),
        "n_cal_paths": len(dataset_3["calibration_keys"]),
        "n_test_paths": len(dataset_3["test_keys"]),
        "calibration_keys": dataset_3["calibration_keys"],
        "test_keys": dataset_3["test_keys"],
    }

    experiments = []
    for arch_name, l, n in archs:
        for steps in steps_list:
            for init_idx in range(1, n_inits + 1):
                model_key = keys[key_cursor]
                train_key = keys[key_cursor + 1]
                key_cursor += 2

                experiments.append((
                    arch_name, int(l), int(n), int(steps), int(init_idx),
                    model_key, train_key,
                    G_cub,
                    train_data,
                    int(batch_size), float(learning_rate),
                    float(loss_alpha), float(loss_beta),
                    strategy_tag,
                    str(out_dir),
                    dataset3_meta,
                ))

    if verbose is None:
        verbose = len(experiments)

    print(
        f"Task 5.2: {len(archs)} archs × {len(steps_list)} steps × {n_inits} inits "
        f"= {len(experiments)} trainings (n_jobs={n_jobs}, backend={backend})"
    )

    results = Parallel(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        max_nbytes=max_nbytes,
    )(
        delayed(_run_single_task5_2_wicub)(exp) for exp in experiments
    )

    return results

def _run_single_task5_3_wf(exp):
    """
    One training run for Task 5.3 (WF):
      (arch, steps, init) with persistence.

    Must be top-level for joblib on Windows (loky).
    """
    import json
    import pickle
    from pathlib import Path

    import jax
    import equinox as eqx
    import klax

    from . import models as tm
    from . import losses as tl

    (
        arch_name, l, n, steps, init_idx,
        model_key, train_key,
        # training arrays
        F_cal, W_cal, P_cal, weights_cal,
        # hyperparams
        batch_size, learning_rate,
        loss_alpha, loss_beta,
        strategy_tag,
        out_dir,
        dataset3_meta,
    ) = exp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Training data format for WeightedSobolevLoss:
        # batch = (F, ((W_true, P_true), weights))
        train_data_WF = (
            F_cal,
            ((W_cal, P_cal), weights_cal),
        )

        loss_fn = tl.WeightedSobolevLoss(alpha=loss_alpha, beta=loss_beta)

        # WF model (F-only input; internal feature construction)
        WF_model = tm.SobolevModel_WF(
            key=model_key,
            input_dim=19,           # internal (F, cofF, detF)
            output_dim="scalar",
            num_hidden_layers=l,
            nodes_per_layer=n,
            activation=jax.nn.softplus,
            is_icnn=True,
            is_ficnn=False,
        )

        trained_model, history = tm.train_model(
            model=WF_model,
            train_data=train_data_WF,
            key=train_key,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
        )

        final_model = klax.finalize(trained_model)

        tag = f"WF_{strategy_tag}_{arch_name}_l{l}_n{n}_steps{steps}_init{init_idx:02d}"
        model_path = out_dir / f"{tag}.eqx"
        hist_path  = out_dir / f"{tag}_history.pkl"
        meta_path  = out_dir / f"{tag}.json"

        # Prefer your helpers; fall back to direct saving if needed.
        try:
            _save_eqx_model(final_model, model_path)  # noqa: F821
            _save_history(history, hist_path)         # noqa: F821
        except Exception:
            eqx.tree_serialise_leaves(str(model_path), final_model)
            with open(hist_path, "wb") as f:
                pickle.dump(history, f)

        meta = {
            "task": "5.3",
            "model_id": "WF",
            "tag": tag,
            "architecture": {"l": l, "n": n, "activation": "softplus"},
            "steps": int(steps),
            "init_idx": int(init_idx),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "loss": "WeightedSobolevLoss",
            "loss_params": {"alpha": float(loss_alpha), "beta": float(loss_beta)},
            "icnn": {"is_icnn": True, "is_ficnn": False},
            "dataset_3": dataset3_meta,
            "saved_model_path": str(model_path),
            "saved_history_path": str(hist_path),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta

    except Exception as e:
        print(f"[Task5.3][WF] FAIL arch={arch_name} steps={steps} init={init_idx:02d}: {e}")
        return {
            "task": "5.3",
            "model_id": "WF",
            "arch": arch_name,
            "l": int(l),
            "n": int(n),
            "steps": int(steps),
            "init_idx": int(init_idx),
            "error": str(e),
        }

def workflow_task_5_3_sweep_wf(
    dataset_3: dict,
    *,
    out_dir: str | Path = "artifacts/task5_3",
    n_inits: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    steps_list=None,
    archs=None,
    loss_alpha: float = 1.0,
    loss_beta: float = 0.0,
    master_key: jrandom.PRNGKey = jrandom.PRNGKey(0),
    # ---- joblib knobs ----
    n_jobs: int = -2,
    backend: str = "loky",
    verbose: int | None = None,
    max_nbytes: str | None = "50M",
):
    """
    Task 5.3: Parallel sweep WF model configurations on Dataset 3 (training only).

    Parallel unit = (arch, steps, init).
    """
    from joblib import Parallel, delayed
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if steps_list is None:
        steps_list = [100_000, 300_000, 500_000]

    if archs is None:
        archs = [
            ("small",  2,  8),
            ("medium", 3, 16),
            ("large",  4, 32),
        ]

    # Unpack dataset_3 (already scaled and properly aligned)
    F_cal = dataset_3["F_cal"]
    W_cal = dataset_3["W_cal"]
    P_cal = dataset_3["P_cal"]
    weights_cal = dataset_3["weights_cal"]

    strategy_tag = f"a{loss_alpha:g}_b{loss_beta:g}"  # e.g. a1_b0

    # Preserve deterministic key assignment like your serial workflow
    total_runs = len(archs) * len(steps_list) * n_inits
    keys = jrandom.split(master_key, total_runs * 2 + 1)
    key_cursor = 1

    # Small meta payload (pickle-friendly)
    dataset3_meta = {
        "path_h5": dataset_3.get("path_h5", None),
        "alpha_scale": float(dataset_3["alpha"]),
        "P_max": float(dataset_3["P_max"]),
        "n_cal_paths": len(dataset_3["calibration_keys"]),
        "n_test_paths": len(dataset_3["test_keys"]),
        "calibration_keys": dataset_3["calibration_keys"],
        "test_keys": dataset_3["test_keys"],
    }

    experiments = []
    for arch_name, l, n in archs:
        for steps in steps_list:
            for init_idx in range(1, n_inits + 1):
                model_key = keys[key_cursor]
                train_key = keys[key_cursor + 1]
                key_cursor += 2

                experiments.append((
                    arch_name, int(l), int(n), int(steps), int(init_idx),
                    model_key, train_key,
                    F_cal, W_cal, P_cal, weights_cal,
                    int(batch_size), float(learning_rate),
                    float(loss_alpha), float(loss_beta),
                    strategy_tag,
                    str(out_dir),
                    dataset3_meta,
                ))

    if verbose is None:
        verbose = len(experiments)

    print(
        f"Task 5.3: {len(archs)} archs × {len(steps_list)} steps × {n_inits} inits "
        f"= {len(experiments)} trainings (n_jobs={n_jobs}, backend={backend})"
    )

    results = Parallel(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        max_nbytes=max_nbytes,
    )(
        delayed(_run_single_task5_3_wf)(exp) for exp in experiments
    )

    return results

def _build_augmented_wf_dataset(F_base, W_base, P_base, weights_base, *, observers, key_seed):
    """
    Build augmented WF dataset using td2.augment_WF_data.
    """
    aug_key = jrandom.PRNGKey(key_seed + observers)
    F_aug, W_aug, P_aug = td2.augment_WF_data(
        F_base, W_base, P_base,
        num_observers=observers,
        key=aug_key
    )
    
    # Weights: original + tiled for rotated copies
    weights_aug = jnp.concatenate(
        [weights_base, jnp.tile(weights_base, (observers,))],
        axis=0
    ).reshape(-1)
    
    return F_aug, W_aug, P_aug, weights_aug


def _run_single_task5_4_wf_augmented(exp):
    """
    One training run for Task 5.4:
      (observer_count, init_idx) for WF on augmented dataset.

    Must be top-level for joblib (Windows loky).
    """
    import json
    import pickle
    from pathlib import Path

    import jax
    import jax.random as jrandom
    import equinox as eqx
    import klax

    from . import models as tm
    from . import losses as tl

    (
        observers,
        init_idx,
        # fixed best config
        best_l, best_n, steps,
        # augmented training data
        F_aug, W_aug, P_aug, weights_aug,
        # hyperparams
        batch_size, learning_rate,
        loss_alpha, loss_beta,
        # seeding
        model_key, train_key,
        # output
        out_dir,
        dataset3_meta,
    ) = exp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Training data format for WF + Sobolev loss
        train_data = (
            F_aug,
            ((W_aug, P_aug), weights_aug),
        )

        loss_fn = tl.WeightedSobolevLoss(alpha=loss_alpha, beta=loss_beta)

        WF_model = tm.SobolevModel_WF(
            key=model_key,
            input_dim=19,
            output_dim="scalar",
            num_hidden_layers=best_l,
            nodes_per_layer=best_n,
            activation=jax.nn.softplus,
            is_icnn=True,
            is_ficnn=False,
        )

        trained_model, history = tm.train_model(
            model=WF_model,
            train_data=train_data,
            key=train_key,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
        )

        final_model = klax.finalize(trained_model)

        tag = (
            f"WF_AUG_obs{observers}"
            f"_l{best_l}_n{best_n}_steps{steps}"
            f"_init{init_idx:02d}"
        )

        model_path = out_dir / f"{tag}.eqx"
        hist_path  = out_dir / f"{tag}_history.pkl"
        meta_path  = out_dir / f"{tag}.json"

        try:
            _save_eqx_model(final_model, model_path)   # noqa: F821
            _save_history(history, hist_path)          # noqa: F821
        except Exception:
            eqx.tree_serialise_leaves(str(model_path), final_model)
            with open(hist_path, "wb") as f:
                pickle.dump(history, f)

        meta = {
            "task": "5.4",
            "model_id": "WF",
            "tag": tag,
            "augmentation": {
                "observers": int(observers),
                "type": "multiscale_rotation_sampling",
            },
            "architecture": {
                "l": int(best_l),
                "n": int(best_n),
                "activation": "softplus",
            },
            "steps": int(steps),
            "init_idx": int(init_idx),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "loss": "WeightedSobolevLoss",
            "loss_params": {"alpha": float(loss_alpha), "beta": float(loss_beta)},
            "icnn": {"is_icnn": True, "is_ficnn": False},
            "dataset_3": dataset3_meta,
            "saved_model_path": str(model_path),
            "saved_history_path": str(hist_path),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta

    except Exception as e:
        print(f"[Task5.4][WF_AUG] FAIL obs={observers} init={init_idx:02d}: {e}")
        return {
            "task": "5.4",
            "model_id": "WF",
            "observers": int(observers),
            "init_idx": int(init_idx),
            "error": str(e),
        }

def workflow_task_5_4_train_wf_augmented(
    dataset_3: dict,
    *,
    best_l: int,
    best_n: int,
    steps: int,
    observers_list=(8, 16, 32, 64),
    n_inits: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    loss_alpha: float = 1.0,
    loss_beta: float = 0.0,
    out_dir: str | Path = "artifacts/task5_4",
    master_key=jrandom.PRNGKey(0),
    aug_key_seed: int = 1234,
    # ---- joblib knobs ----
    n_jobs: int = -2,
    backend: str = "loky",
    verbose: int | None = None,
    max_nbytes: str | None = "50M",
):
    """
    Task 5.4: Parallel WF training on augmented datasets.

    Parallel unit = (observer setting, init).
    """
    from pathlib import Path
    from joblib import Parallel, delayed
    import jax.numpy as jnp
    import jax.random as jrandom

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Unpack base dataset
    F_base = dataset_3["F_cal"]
    W_base = dataset_3["W_cal"]
    P_base = dataset_3["P_cal"]
    weights_base = dataset_3["weights_cal"]

    # Metadata payload (small + pickle-safe)
    dataset3_meta = {
        "path_h5": dataset_3.get("path_h5", None),
        "alpha_scale": float(dataset_3["alpha"]),
        "P_max": float(dataset_3["P_max"]),
        "calibration_keys": dataset_3["calibration_keys"],
        "test_keys": dataset_3["test_keys"],
    }

    # Deterministic key splitting (same philosophy as other tasks)
    total_runs = len(observers_list) * n_inits
    keys = jrandom.split(master_key, total_runs * 2 + 1)
    key_cursor = 1

    experiments = []

    for obs in observers_list:
        # --- build augmented dataset exactly like serial workflow ---
        F_aug, W_aug, P_aug, weights_aug = _build_augmented_wf_dataset(  # noqa: F821
            F_base,
            W_base,
            P_base,
            weights_base,
            observers=obs,
            key_seed=aug_key_seed,
        )

        for init_idx in range(1, n_inits + 1):
            model_key = keys[key_cursor]
            train_key = keys[key_cursor + 1]
            key_cursor += 2

            experiments.append((
                int(obs),
                int(init_idx),
                int(best_l), int(best_n), int(steps),
                F_aug, W_aug, P_aug, weights_aug,
                int(batch_size), float(learning_rate),
                float(loss_alpha), float(loss_beta),
                model_key, train_key,
                str(out_dir),
                dataset3_meta,
            ))

    if verbose is None:
        verbose = len(experiments)

    print(
        f"Task 5.4: {len(observers_list)} observer settings × {n_inits} inits "
        f"= {len(experiments)} trainings (n_jobs={n_jobs}, backend={backend})"
    )

    results = Parallel(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        max_nbytes=max_nbytes,
    )(
        delayed(_run_single_task5_4_wf_augmented)(exp) for exp in experiments
    )

    return results
