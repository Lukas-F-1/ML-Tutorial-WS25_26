"""Experiment runner for viscoelasticity models."""

from dataclasses import dataclass, field
from typing import Any
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import klax
import numpy as np

from .configs import ExperimentConfig, ModelType, MATERIAL_PARAMS
from .metrics import compute_all_metrics
from . import data as td
from . import models as tm


# =============================================================================
# Result Data Structures
# =============================================================================
@dataclass
class ModelResult:
    """Results for a single model on a single experiment."""

    model_type: ModelType
    experiment_name: str

    # Training info
    train_time: float = 0.0
    train_steps: int = 0
    final_loss: float = 0.0

    # Metrics per test case: {loadcase_str: metrics_dict}
    harmonic_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    relaxation_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    # Predictions (for plotting)
    predictions: dict[str, np.ndarray] = field(default_factory=dict)

    # Trained model (optional, for further analysis)
    model: Any = None


@dataclass
class ExperimentResult:
    """Results for a complete experiment (all models)."""

    config: ExperimentConfig
    model_results: dict[ModelType, ModelResult] = field(default_factory=dict)

    def get_summary_df(self):
        """Return a summary as pandas DataFrame (if pandas available)."""
        try:
            import pandas as pd

            rows = []
            for model_type, result in self.model_results.items():
                for loadcase, metrics in result.harmonic_metrics.items():
                    row = {
                        "model": model_type,
                        "loadcase": loadcase,
                        "type": "harmonic",
                        **metrics,
                    }
                    rows.append(row)
                for loadcase, metrics in result.relaxation_metrics.items():
                    row = {
                        "model": model_type,
                        "loadcase": loadcase,
                        "type": "relaxation",
                        **metrics,
                    }
                    rows.append(row)
            return pd.DataFrame(rows)
        except ImportError:
            return None


# =============================================================================
# Data Generation (extended for multi-period)
# =============================================================================
def generate_training_data(
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate training data based on experiment config."""
    E_infty = MATERIAL_PARAMS["E_infty"]
    E = MATERIAL_PARAMS["E"]
    eta = MATERIAL_PARAMS["eta"]

    omegas = [lc[1] for lc in config.train_loadcases]
    As = [lc[0] for lc in config.train_loadcases]

    # Generate data for specified number of periods
    if config.n_periods == 1:
        return td.generate_data_harmonic(E_infty, E, eta, config.n_timesteps, omegas, As)
    else:
        # Multi-period: concatenate multiple periods
        all_eps, all_eps_dot, all_sig, all_dts = [], [], [], []
        for _ in range(config.n_periods):
            eps, eps_dot, sig, dts = td.generate_data_harmonic(
                E_infty, E, eta, config.n_timesteps, omegas, As
            )
            all_eps.append(eps)
            all_eps_dot.append(eps_dot)
            all_sig.append(sig)
            all_dts.append(dts)

        return (
            np.concatenate(all_eps, axis=1),
            np.concatenate(all_eps_dot, axis=1),
            np.concatenate(all_sig, axis=1),
            np.concatenate(all_dts, axis=1),
        )


def generate_test_data(
    config: ExperimentConfig, test_type: str = "harmonic"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Generate test data based on experiment config."""
    E_infty = MATERIAL_PARAMS["E_infty"]
    E = MATERIAL_PARAMS["E"]
    eta = MATERIAL_PARAMS["eta"]

    omegas = [lc[1] for lc in config.test_loadcases]
    As = [lc[0] for lc in config.test_loadcases]

    if test_type == "harmonic":
        eps, eps_dot, sig, dts = td.generate_data_harmonic(
            E_infty, E, eta, config.n_timesteps, omegas, As
        )
    else:  # relaxation
        eps, eps_dot, sig, dts = td.generate_data_relaxation(
            E_infty, E, eta, config.n_timesteps, omegas, As
        )

    return eps, eps_dot, sig, dts, config.test_loadcases


# =============================================================================
# Model Building
# =============================================================================
def build_model(model_type: ModelType, key: jrandom.PRNGKey):
    """Build a model based on type."""
    E_infty = MATERIAL_PARAMS["E_infty"]
    E = MATERIAL_PARAMS["E"]
    eta = MATERIAL_PARAMS["eta"]

    if model_type == "simple_rnn":
        return tm.build(key=key)
    elif model_type == "maxwell":
        return tm.build_maxwell(E_infty=E_infty, E_val=E, eta=eta)
    elif model_type == "maxwell_nn":
        return tm.build_maxwell_nn(key=key, E_infty=E_infty, E_val=E)
    elif model_type == "gsm":
        return tm.build_gsm(key=key, g=1.0 / eta)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Training
# =============================================================================
def train_model(
    model,
    train_data: tuple[np.ndarray, np.ndarray],
    target: np.ndarray,
    config: ExperimentConfig,
    key: jrandom.PRNGKey,
) -> tuple[Any, float, float]:
    """Train a model and return (trained_model, train_time, final_loss)."""
    eps, dts = train_data
    sig = target

    t_start = time.time()

    model, history = klax.fit(
        model,
        ((eps, dts), sig),
        batch_axis=0,
        steps=config.train_steps,
        history=klax.HistoryCallback(log_every=config.log_every),
        key=key,
    )

    train_time = time.time() - t_start
    final_loss = float(history.history["loss"][-1]) if history.history["loss"] else 0.0

    return model, train_time, final_loss


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_model(
    model, test_data: tuple[np.ndarray, np.ndarray], target: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate a model and return (predictions, metrics)."""
    eps, dts = test_data

    # Finalize model (unwrap wrappers, apply constraints)
    model_final = klax.finalize(model)

    # Predict
    sig_pred = jax.vmap(model_final)((eps, dts))
    sig_pred = np.array(sig_pred)

    # Compute metrics
    metrics = compute_all_metrics(target, sig_pred)

    return sig_pred, metrics


# =============================================================================
# Run Single Experiment
# =============================================================================
def run_experiment(
    config: ExperimentConfig,
    seed: int = 42,
    verbose: bool = True,
) -> ExperimentResult:
    """Run a complete experiment with all specified models."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'='*60}")

    result = ExperimentResult(config=config)

    # Generate training data
    eps_train, _, sig_train, dts_train = generate_training_data(config)

    # Generate test data
    eps_test_h, _, sig_test_h, dts_test_h, loadcases = generate_test_data(
        config, "harmonic"
    )

    if config.test_relaxation:
        eps_test_r, _, sig_test_r, dts_test_r, _ = generate_test_data(
            config, "relaxation"
        )

    # Run each model
    for model_type in config.models:
        if verbose:
            print(f"\n--- {model_type} ---")

        model_result = ModelResult(
            model_type=model_type, experiment_name=config.name
        )

        # Create random key
        key = jrandom.PRNGKey(seed)
        keys = jrandom.split(key, 2)

        # Build model
        model = build_model(model_type, keys[0])

        # Train (if trainable)
        if model_type != "maxwell":
            if verbose:
                print(f"Training for {config.train_steps} steps...")

            model, train_time, final_loss = train_model(
                model,
                (eps_train, dts_train),
                sig_train,
                config,
                keys[1],
            )
            model_result.train_time = train_time
            model_result.train_steps = config.train_steps
            model_result.final_loss = final_loss

            if verbose:
                print(f"Training time: {train_time:.2f}s, Final loss: {final_loss:.6f}")
        else:
            if verbose:
                print("Analytical model - no training needed")

        # Evaluate on harmonic test cases
        sig_pred_h, _ = evaluate_model(model, (eps_test_h, dts_test_h), sig_test_h)
        model_result.predictions["harmonic"] = sig_pred_h

        # Compute metrics per loadcase
        for i, (A, omega) in enumerate(loadcases):
            lc_str = f"A={A},w={omega}"
            metrics = compute_all_metrics(sig_test_h[i], sig_pred_h[i])
            model_result.harmonic_metrics[lc_str] = metrics

            if verbose:
                print(f"  {lc_str}: RMSE={metrics['rmse']:.4f}, R²={metrics['r_squared']:.4f}")

        # Evaluate on relaxation test cases
        if config.test_relaxation:
            sig_pred_r, _ = evaluate_model(model, (eps_test_r, dts_test_r), sig_test_r)
            model_result.predictions["relaxation"] = sig_pred_r

            for i, (A, omega) in enumerate(loadcases):
                lc_str = f"A={A},w={omega}"
                metrics = compute_all_metrics(sig_test_r[i], sig_pred_r[i])
                model_result.relaxation_metrics[lc_str] = metrics

        # Store model
        model_result.model = model

        result.model_results[model_type] = model_result

    return result


# =============================================================================
# Run Sweep (Multiple Experiments)
# =============================================================================
def run_sweep(
    experiments: list[ExperimentConfig],
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, ExperimentResult]:
    """Run multiple experiments and return all results."""
    results = {}

    for i, config in enumerate(experiments):
        if verbose:
            print(f"\n[{i+1}/{len(experiments)}] Running experiment: {config.name}")

        result = run_experiment(config, seed=seed, verbose=verbose)
        results[config.name] = result

    if verbose:
        print(f"\n{'='*60}")
        print(f"Sweep complete! Ran {len(experiments)} experiments.")
        print(f"{'='*60}")

    return results
