"""Plotting utilities."""

from __future__ import annotations
import json
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .experiments import ExperimentResult

from . import data as td
from . import models as tm
from . import storage
from .configs import MATERIAL_PARAMS
import jax
import jax.random as jrandom
import klax


# Colors for loadcases
LOADCASE_COLORS = np.array(
    [
        [(194 / 255, 76 / 255, 76 / 255)],
        [(246 / 255, 163 / 255, 21 / 255)],
        [(67 / 255, 83 / 255, 132 / 255)],
        [(22 / 255, 164 / 255, 138 / 255)],
        [(104 / 255, 143 / 255, 198 / 255)],
    ]
)

# Colors for models
MODEL_COLORS = {
    "simple_rnn": "#c24c4c",  # red
    "maxwell": "#436384",  # blue
    "maxwell_nn": "#16a48a",  # teal
    "gsm": "#f6a315",  # orange
}

MODEL_LABELS = {
    "simple_rnn": "Simple RNN",
    "maxwell": "Maxwell (analytical)",
    "maxwell_nn": "Maxwell + NN",
    "gsm": "GSM",
}

# Keep old name for backwards compatibility
colors = LOADCASE_COLORS


def plot_data(eps, eps_dot, sig, omegas, As):
    n = len(eps[0])
    ns = np.linspace(0, 2 * np.pi, n)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Data")

    ax = axs[0, 0]
    for i in range(len(eps)):
        ax.plot(
            ns,
            sig[i],
            label="$\\omega$: %.2f, $A$: %.2f" % (omegas[i], As[i]),
            color=colors[i],
            linestyle="--",
        )
    ax.set_xlim([0, 2 * np.pi])
    ax.set_ylabel("stress $\\sigma$")
    ax.set_xlabel("time $t$")
    ax.legend()

    ax = axs[0, 1]
    for i in range(len(eps)):
        ax.plot(eps[i], sig[i], color=colors[i], linestyle="--")
    ax.set_xlabel("strain $\\varepsilon$")
    ax.set_ylabel("stress $\\sigma$")

    ax = axs[1, 0]
    for i in range(len(eps)):
        ax.plot(ns, eps[i], color=colors[i], linestyle="--")
    ax.set_xlim([0, 2 * np.pi])
    ax.set_xlabel("time $t$")
    ax.set_ylabel("strain $\\varepsilon$")

    ax = axs[1, 1]
    for i in range(len(eps)):
        ax.plot(ns, eps_dot[i], color=colors[i], linestyle="--")
    ax.set_xlim([0, 2 * np.pi])
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"strain rate $\.{\varepsilon}$")

    fig.tight_layout()
    plt.show()


def plot_model_pred(eps, sig, sig_m, omegas, As, title=None):
    n = len(eps[0])
    ns = np.linspace(0, 2 * np.pi, n)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    if title:
        fig.suptitle(title, fontsize=11)
    else:
        fig.suptitle("Data: dashed line, model prediction: continuous line")

    ax = axs[0]
    for i in range(len(eps)):
        ax.plot(
            ns,
            sig[i],
            label="$\\omega$: %.2f, $A$: %.2f" % (omegas[i], As[i]),
            linestyle="--",
            color=colors[i],
        )
        ax.plot(ns, sig_m[i], color=colors[i])
    ax.set_xlim([0, 2 * np.pi])
    ax.set_ylabel("stress $\\sigma$")
    ax.set_xlabel("time $t$")
    ax.legend()

    ax = axs[1]
    for i in range(len(eps)):
        ax.plot(eps[i], sig[i], linestyle="--", color=colors[i])
        ax.plot(eps[i], sig_m[i], color=colors[i])
    ax.set_xlabel("strain $\\varepsilon$")
    ax.set_ylabel("stress $\\sigma$")

    fig.tight_layout()
    plt.show()


# =============================================================================
# Comparison Plots for Experiments
# =============================================================================


def plot_model_comparison(
    result: ExperimentResult,
    test_type: str = "harmonic",
    figsize: tuple[int, int] = (14, 5),
) -> None:
    """Plot all models side by side for an experiment.

    Args:
        result: ExperimentResult from run_experiment()
        test_type: "harmonic" or "relaxation"
        figsize: Figure size
    """
    config = result.config
    n_models = len(result.model_results)
    loadcases = config.test_loadcases

    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        f"{config.name}: {config.description}\n({test_type} test)",
        fontsize=12,
    )

    # Get test data for plotting
    from . import data as td
    from .configs import MATERIAL_PARAMS

    omegas = [lc[1] for lc in loadcases]
    As = [lc[0] for lc in loadcases]

    if test_type == "harmonic":
        eps, _, sig, _ = td.generate_data_harmonic(
            MATERIAL_PARAMS["E_infty"],
            MATERIAL_PARAMS["E"],
            MATERIAL_PARAMS["eta"],
            config.n_timesteps,
            omegas,
            As,
        )
    else:
        eps, _, sig, _ = td.generate_data_relaxation(
            MATERIAL_PARAMS["E_infty"],
            MATERIAL_PARAMS["E"],
            MATERIAL_PARAMS["eta"],
            config.n_timesteps,
            omegas,
            As,
        )

    n_points = len(eps[0])
    ts = np.linspace(0, 2 * np.pi, n_points)

    for ax_idx, (model_type, model_result) in enumerate(result.model_results.items()):
        ax = axes[ax_idx]
        sig_pred = model_result.predictions.get(test_type)

        if sig_pred is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Plot each loadcase
        for i, (A, omega) in enumerate(loadcases):
            color = LOADCASE_COLORS[i % len(LOADCASE_COLORS)].flatten()
            ax.plot(ts, sig[i], "--", color=color, alpha=0.7, linewidth=1.5)
            ax.plot(
                ts,
                sig_pred[i],
                "-",
                color=color,
                linewidth=2,
                label=f"A={A}, ω={omega}",
            )

        # Get metrics for title
        metrics = model_result.harmonic_metrics if test_type == "harmonic" else model_result.relaxation_metrics
        avg_rmse = np.mean([m["rmse"] for m in metrics.values()])
        avg_r2 = np.mean([m["r_squared"] for m in metrics.values()])

        ax.set_title(f"{MODEL_LABELS.get(model_type, model_type)}\nRMSE={avg_rmse:.4f}, R²={avg_r2:.4f}")
        ax.set_xlabel("time $t$")
        ax.set_ylabel("stress $\\sigma$")
        ax.set_xlim([0, 2 * np.pi])

        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()


def plot_metrics_comparison(
    results: dict[str, ExperimentResult],
    metric: str = "rmse",
    test_type: str = "harmonic",
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """Bar chart comparing a metric across experiments and models.

    Args:
        results: Dict of experiment_name -> ExperimentResult
        metric: Which metric to plot ("rmse", "r_squared", "mae", etc.)
        test_type: "harmonic" or "relaxation"
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    experiment_names = list(results.keys())
    n_experiments = len(experiment_names)

    # Collect all model types across experiments
    all_models = set()
    for result in results.values():
        all_models.update(result.model_results.keys())
    all_models = sorted(all_models)
    n_models = len(all_models)

    # Bar positions
    bar_width = 0.8 / n_models
    x = np.arange(n_experiments)

    for i, model_type in enumerate(all_models):
        values = []
        for exp_name in experiment_names:
            result = results[exp_name]
            if model_type in result.model_results:
                model_result = result.model_results[model_type]
                metrics_dict = (
                    model_result.harmonic_metrics
                    if test_type == "harmonic"
                    else model_result.relaxation_metrics
                )
                # Average across loadcases
                avg_value = np.mean([m[metric] for m in metrics_dict.values()])
                values.append(avg_value)
            else:
                values.append(0)

        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            bar_width,
            label=MODEL_LABELS.get(model_type, model_type),
            color=MODEL_COLORS.get(model_type, "gray"),
        )

    ax.set_xlabel("Experiment")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} Comparison ({test_type})")
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_metrics_heatmap(
    result: ExperimentResult,
    metric: str = "rmse",
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Heatmap of metric values: models x loadcases.

    Args:
        result: ExperimentResult from run_experiment()
        metric: Which metric to plot
        figsize: Figure size
    """
    models = list(result.model_results.keys())
    loadcases = list(result.config.test_loadcases)
    loadcase_strs = [f"A={A},w={w}" for A, w in loadcases]

    # Build matrix
    data_h = np.zeros((len(models), len(loadcases)))
    data_r = np.zeros((len(models), len(loadcases)))

    for i, model_type in enumerate(models):
        model_result = result.model_results[model_type]
        for j, lc_str in enumerate(loadcase_strs):
            if lc_str in model_result.harmonic_metrics:
                data_h[i, j] = model_result.harmonic_metrics[lc_str][metric]
            if lc_str in model_result.relaxation_metrics:
                data_r[i, j] = model_result.relaxation_metrics[lc_str][metric]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, data, title in zip(axes, [data_h, data_r], ["Harmonic", "Relaxation"]):
        im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(loadcase_strs)))
        ax.set_xticklabels(loadcase_strs, rotation=45, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models])
        ax.set_title(f"{title} - {metric.upper()}")

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(loadcase_strs)):
                text = ax.text(
                    j, i, f"{data[i, j]:.4f}",
                    ha="center", va="center", color="black", fontsize=9
                )

        fig.colorbar(im, ax=ax)

    fig.suptitle(f"{result.config.name}: {result.config.description}")
    fig.tight_layout()
    plt.show()


def print_results_table(
    results: dict[str, ExperimentResult],
    metric: str = "rmse",
    test_type: str = "harmonic",
) -> None:
    """Print a formatted table of results.

    Args:
        results: Dict of experiment_name -> ExperimentResult
        metric: Which metric to show
        test_type: "harmonic" or "relaxation"
    """
    # Collect all models
    all_models = set()
    for result in results.values():
        all_models.update(result.model_results.keys())
    all_models = sorted(all_models)

    # Header
    header = f"{'Experiment':<25}" + "".join([f"{MODEL_LABELS.get(m, m):<18}" for m in all_models])
    print("\n" + "=" * len(header))
    print(f"{metric.upper()} ({test_type})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Rows
    for exp_name, result in results.items():
        row = f"{exp_name:<25}"
        for model_type in all_models:
            if model_type in result.model_results:
                model_result = result.model_results[model_type]
                metrics_dict = (
                    model_result.harmonic_metrics
                    if test_type == "harmonic"
                    else model_result.relaxation_metrics
                )
                avg_value = np.mean([m[metric] for m in metrics_dict.values()])
                row += f"{avg_value:<18.4f}"
            else:
                row += f"{'-':<18}"
        print(row)

    print("=" * len(header) + "\n")

# =============================================================================
# Data Generation Helper
# =============================================================================

def _generate_test_data(n_timesteps, omegas, As, test_type="harmonic", noise_std_rel=0.0):
    """Generate test data, optionally with noisy eps.

    Returns: (eps, sig_true, dts)
    """
    mp = MATERIAL_PARAMS
    if test_type == "harmonic":
        if noise_std_rel > 0:
            eps, _, sig, dts = td.generate_data_harmonic_noisy_eps(
                mp["E_infty"], mp["E"], mp["eta"], n_timesteps, omegas, As,
                noise_std_rel=noise_std_rel, seed=0, recompute_eps_dot_from_noisy=False)
        else:
            eps, _, sig, dts = td.generate_data_harmonic(
                mp["E_infty"], mp["E"], mp["eta"], n_timesteps, omegas, As)
    else:  # relaxation
        if noise_std_rel > 0:
            eps, _, sig, dts = td.generate_data_relaxation_noisy_eps(
                mp["E_infty"], mp["E"], mp["eta"], n_timesteps, omegas, As,
                noise_std_rel=noise_std_rel, seed=0, recompute_eps_dot_from_noisy=False)
        else:
            eps, _, sig, dts = td.generate_data_relaxation(
                mp["E_infty"], mp["E"], mp["eta"], n_timesteps, omegas, As)
    return eps, sig, dts


# =============================================================================
# Single Model Plotting
# =============================================================================

def find_latest(pattern: str, steps=None, search_dirs=None, n_timesteps=None) -> str:
    """Find the latest .eqx file matching a pattern.

    Args:
        pattern: Substring to match in filenames (e.g. "omega_3", "gsm__amp_4__seed_0")
        steps: Optional filter for training steps (e.g. 50000, 100000, 250000)
        search_dirs: List of directories to search. Default: all known artifact dirs
        n_timesteps: Optional filter for number of timesteps (e.g. 50, 100, 200, 400)

    Returns:
        Path to the latest matching .eqx file (sorted by timestamp in filename)
    """
    from pathlib import Path

    if search_dirs is None:
        search_dirs = ["artifacts", "artifacts/gsm_experiments", "artifacts/rnn_experiments", "artifacts/maxwell_nn_experiments"]

    matches = []
    for d in search_dirs:
        p = Path(d)
        if p.exists():
            matches.extend([f for f in p.glob("*.eqx") if pattern in f.name])

    # Filter nach steps falls angegeben
    if steps is not None:
        steps_str = f"{steps}steps"
        matches = [f for f in matches if steps_str in f.name]

    # Filter nach n_timesteps falls angegeben
    if n_timesteps is not None:
        ts_str = f"__{n_timesteps}ts__"
        matches = [f for f in matches if ts_str in f.name]

    if not matches:
        info = f"Pattern='{pattern}'"
        if steps is not None:
            info += f", steps={steps}"
        if n_timesteps is not None:
            info += f", n_timesteps={n_timesteps}"
        print(f"Keine .eqx Dateien gefunden mit {info}")
        print(f"Durchsuchte Ordner: {search_dirs}")
        return None

    # Sortiere nach Dateiname (Timestamp ist am Ende → lexikographisch sortierbar)
    matches.sort(key=lambda f: f.name)
    latest = matches[-1]
    print(f"Gefunden: {latest}")
    return str(latest)


def plot_latest(pattern: str, steps=None, test_loadcases=None, search_dirs=None,
                seeds=None, noise_std_rel=0.0, n_timesteps=None):
    """Find the latest model matching a pattern and plot it.

    Args:
        pattern: Substring to match (e.g. "omega_3", "amp_4__seed_0", "maxwell_nn")
        steps: Optional filter for training steps (e.g. 50000, 150000, 250000)
        test_loadcases: List of (A, omega) tuples to test on. Default: [(1,1), (1,2), (1,3)]
        search_dirs: Optional list of directories to search
        seeds: List of seed indices to plot overlaid in one figure (e.g. [0,2,4])
               Use seeds=[0,1,2,3,4] for all 5 seeds.
        noise_std_rel: Relative noise std on eps (e.g. 0.02 = 2%). Default: 0 (clean)
        n_timesteps: Optional filter for number of timesteps (e.g. 50, 100, 200, 400)

    Examples:
        plot_latest("omega_3__seed_0")                              # single seed
        plot_latest("omega_3__seed_0", noise_std_rel=0.02)          # with 2% noise
        plot_latest("omega_3", seeds=[0,1,2,3,4])                   # all 5 seeds overlaid
        plot_latest("gsm__mixed_4", seeds=[0,1,2], n_timesteps=50)  # timestep study
    """
    if seeds is not None:
        _plot_all_seeds(pattern, steps=steps, seeds=seeds,
                        test_loadcases=test_loadcases, search_dirs=search_dirs,
                        noise_std_rel=noise_std_rel, n_timesteps=n_timesteps)
    else:
        filename = find_latest(pattern, steps=steps, search_dirs=search_dirs,
                               n_timesteps=n_timesteps)
        if filename is not None:
            plot_saved_model(filename, test_loadcases=test_loadcases,
                            noise_std_rel=noise_std_rel)


# Best seeds per config (from visual inspection of all_seeds plots)
BEST_SEEDS_GSM = {
    "omega_1": 0,
    "omega_2": 2,
    "omega_3": 0,
    "omega_4": 1,
    "amp_2":   2,
    "amp_3":   0,
    "amp_4":   0,
    "mixed_2": 4,
    "mixed_4": 1,
}

BEST_SEEDS_RNN = {
    "omega_1": 3,
    "omega_2": 0,
    "omega_3": 2,
    "omega_4": 1,
    "amp_2":   1,  # alle seeds schlecht
    "amp_3":   0,  # nicht sehr gut
    "amp_4":   0,
    "mixed_2": 0,
    "mixed_4": 4,
}

BEST_SEEDS_MAXWELL_NN = {
    "omega_1": 1,
    "omega_2": 2,
    "omega_3": 0,
    "omega_4": 0,
    "amp_2":   3,
    "amp_3":   3,
    "amp_4":   0,
    "mixed_2": 2,
    "mixed_4": 3,
}

# Best seeds for timestep study (model_type -> n_timesteps -> seed)
BEST_SEEDS_TIMESTEP_STUDY = {
    "gsm": {50: 1, 100: 2, 200: 0, 400: 1},
    "maxwell_nn": {50: 0, 100: 2, 200: 1, 400: 1},
    "simple_rnn": {50: 2, 100: 0, 200: 0, 400: 2},
}

# Default search dirs per model type
_SEARCH_DIRS = {
    "gsm":        ["artifacts", "artifacts/gsm_experiments"],
    "simple_rnn": ["artifacts", "artifacts/rnn_experiments"],
    "maxwell_nn": ["artifacts", "artifacts/maxwell_nn_experiments"],
}

def _get_best_seeds(model_type="gsm"):
    if model_type == "simple_rnn":
        return BEST_SEEDS_RNN
    elif model_type == "maxwell_nn":
        return BEST_SEEDS_MAXWELL_NN
    return BEST_SEEDS_GSM

def _get_search_dirs(model_type="gsm", search_dirs=None):
    if search_dirs is not None:
        return search_dirs
    return _SEARCH_DIRS.get(model_type, ["artifacts"])


def plot_best(configs=None, steps=250000, test_loadcases=None, search_dirs=None,
              noise_std_rel=0.0, model_type="gsm", n_test_timesteps=None):
    """Plot best seed of each config overlaid in one figure for comparison.

    All configs are shown in the same plot with different colors.
    Ground truth is black dashed.

    Args:
        configs: List of config names or None for all configs
        steps: Training steps filter (default: 250000)
        test_loadcases: List of (A, omega) tuples. Default: [(1,1)]
        search_dirs: Optional list of directories to search
        noise_std_rel: Relative noise std on eps (e.g. 0.02 = 2%). Default: 0 (clean)
        model_type: "gsm", "simple_rnn", or "maxwell_nn" (selects best seeds + search dirs)

    Examples:
        plot_best()                                          # GSM best seeds
        plot_best(model_type="simple_rnn")                   # RNN best seeds
        plot_best(model_type="maxwell_nn")                   # Maxwell NN best seeds
        plot_best(["omega_2", "omega_4"], model_type="simple_rnn")
    """
    best_seeds = _get_best_seeds(model_type)
    search_dirs = _get_search_dirs(model_type, search_dirs)

    if configs is None:
        configs = list(best_seeds.keys())
    if test_loadcases is None:
        test_loadcases = [(1.0, 1.0)]

    As = [lc[0] for lc in test_loadcases]
    omegas = [lc[1] for lc in test_loadcases]

    # Collect model files
    model_files = []
    for config in configs:
        seed = best_seeds.get(config)
        if seed is None:
            print(f"Kein best seed definiert für '{config}', überspringe...")
            continue
        pattern = f"{config}__seed_{seed}"
        f = find_latest(pattern, steps=steps, search_dirs=search_dirs)
        if f is not None:
            model_files.append((config, seed, f))

    if not model_files:
        print("Keine Modelle gefunden.")
        return

    # Parse n_timesteps and file_model_type from first file
    name_only = str(model_files[0][2]).split("/")[-1].split("\\")[-1]
    parts = name_only.replace(".eqx", "").split("__")
    if len(parts) == 6:
        file_model_type = parts[0]
        n_timesteps = int(parts[4].replace("ts", ""))
    elif len(parts) == 5:
        file_model_type = parts[0]
        n_timesteps = int(parts[3].replace("ts", ""))
    else:
        print(f"Unbekanntes Format: {name_only}")
        return

    # Build model template
    key = jrandom.PRNGKey(0)
    if file_model_type == "gsm":
        model_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
    elif file_model_type == "simple_rnn":
        model_template = tm.build(key=key)
    elif file_model_type == "maxwell_nn":
        model_template = tm.build_maxwell_nn(
            key=key, E_infty=MATERIAL_PARAMS["E_infty"], E_val=MATERIAL_PARAMS["E"])
    else:
        print(f"Unbekannter Modelltyp: {file_model_type}")
        return

    # Generate test data (use n_test_timesteps if provided, else n_timesteps from model file)
    n_ts_test = n_test_timesteps if n_test_timesteps is not None else n_timesteps
    eps_h, sig_h, dts_h = _generate_test_data(n_ts_test, omegas, As, "harmonic", noise_std_rel)
    eps_r, sig_r, dts_r = _generate_test_data(n_ts_test, omegas, As, "relaxation", noise_std_rel)

    n_pts = len(eps_h[0])
    ns = np.linspace(0, 2 * np.pi, n_pts)
    n_lc = len(test_loadcases)
    cmap = plt.cm.tab10

    # Title
    tc_str = ", ".join([f"(A={a},ω={w})" for a, w in test_loadcases])
    noise_str = f", noise={noise_std_rel:.0%}" if noise_std_rel > 0 else ""
    ts_str = f", test_ts={n_ts_test}" if n_test_timesteps is not None else ""
    model_label = MODEL_LABELS.get(file_model_type, file_model_type.upper())

    # --- Harmonic Plot ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_label} Best Seeds Comparison — Harmonic — Test: {tc_str}{noise_str}{ts_str}", fontsize=11)

    for i in range(n_lc):
        lbl = f"GT (A={As[i]},ω={omegas[i]})" if n_lc > 1 else "Ground Truth"
        axs[0].plot(ns, sig_h[i], linestyle=":", color="black", linewidth=2, label=lbl if i == 0 else None)
        axs[1].plot(eps_h[i], sig_h[i], linestyle=":", color="black", linewidth=2)

    for idx, (config, seed, filepath) in enumerate(model_files):
        model = storage.load_model(filepath, model_template)
        model = klax.finalize(model)
        sig_pred = jax.vmap(model)((eps_h, dts_h))
        c = cmap(idx % 10)
        for i in range(n_lc):
            label = f"{config} (s{seed})" if i == 0 else None
            axs[0].plot(ns, sig_pred[i], color=c, alpha=0.8, label=label)
            axs[1].plot(eps_h[i], sig_pred[i], color=c, alpha=0.8)

    axs[0].set_xlim([0, 2 * np.pi])
    axs[0].set_ylabel("stress $\\sigma$")
    axs[0].set_xlabel("time $t$")
    axs[0].legend(fontsize=7, loc="best")
    axs[1].set_xlabel("strain $\\varepsilon$")
    axs[1].set_ylabel("stress $\\sigma$")
    fig.tight_layout()
    plt.show()

    # --- Relaxation Plot ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{model_label} Best Seeds Comparison — Relaxation — Test: {tc_str}{noise_str}{ts_str}", fontsize=11)

    for i in range(n_lc):
        lbl = f"GT (A={As[i]},ω={omegas[i]})" if n_lc > 1 else "Ground Truth"
        axs[0].plot(ns, sig_r[i], linestyle=":", color="black", linewidth=2, label=lbl if i == 0 else None)
        axs[1].plot(eps_r[i], sig_r[i], linestyle=":", color="black", linewidth=2)

    for idx, (config, seed, filepath) in enumerate(model_files):
        model = storage.load_model(filepath, model_template)
        model = klax.finalize(model)
        sig_pred = jax.vmap(model)((eps_r, dts_r))
        c = cmap(idx % 10)
        for i in range(n_lc):
            label = f"{config} (s{seed})" if i == 0 else None
            axs[0].plot(ns, sig_pred[i], color=c, alpha=0.8, label=label)
            axs[1].plot(eps_r[i], sig_pred[i], color=c, alpha=0.8)

    axs[0].set_xlim([0, 2 * np.pi])
    axs[0].set_ylabel("stress $\\sigma$")
    axs[0].set_xlabel("time $t$")
    axs[0].legend(fontsize=7, loc="best")
    axs[1].set_xlabel("strain $\\varepsilon$")
    axs[1].set_ylabel("stress $\\sigma$")
    fig.tight_layout()
    plt.show()


def plot_heatmaps(configs=None, steps=250000, test_omegas=None, test_As=None,
                  test_type="harmonic", log=False, normalize=False, noise_std_rel=0.0,
                  search_dirs=None, model_type="gsm", n_test_timesteps=None):
    """Plot RMSE heatmaps for each config's best seed over a grid of (A, omega) test cases.

    Args:
        configs: List of config names or None for all in BEST_SEEDS_{GSM/RNN}
        steps: Training steps filter (default: 250000)
        test_omegas: List of omega values for the grid. Default: range(1,21)
        test_As: List of A values for the grid. Default: range(1,21)
        test_type: "harmonic" or "relaxation"
        log: If True, use logarithmic color scale
        normalize: If True, use NRMSE (RMSE / std(sigma_true)) instead of RMSE
        noise_std_rel: Relative noise std on eps (e.g. 0.02 = 2%). Default: 0 (clean)
        search_dirs: Optional list of directories to search
        model_type: "gsm", "simple_rnn", or "maxwell_nn" (selects best seeds + search dirs)

    Examples:
        plot_heatmaps()
        plot_heatmaps(["omega_1", "omega_4", "mixed_4"])
        plot_heatmaps(log=True)
        plot_heatmaps(normalize=True, log=True)
        plot_heatmaps(noise_std_rel=0.02)
        plot_heatmaps(test_omegas=range(1,11), test_As=range(1,11))
        plot_heatmaps(model_type="simple_rnn")
        plot_heatmaps(model_type="maxwell_nn")
        plot_heatmaps(n_test_timesteps=200)
    """
    from matplotlib.colors import LogNorm

    best_seeds = _get_best_seeds(model_type)
    search_dirs = _get_search_dirs(model_type, search_dirs)

    if configs is None:
        configs = list(best_seeds.keys())
    if test_omegas is None:
        test_omegas = list(range(1, 21))
    if test_As is None:
        test_As = list(range(1, 21))

    # Collect model files
    model_files = []
    for config in configs:
        seed = best_seeds.get(config)
        if seed is None:
            print(f"Kein best seed definiert für '{config}', überspringe...")
            continue
        pattern = f"{config}__seed_{seed}"
        f = find_latest(pattern, steps=steps, search_dirs=search_dirs)
        if f is not None:
            model_files.append((config, seed, f))

    if not model_files:
        print("Keine Modelle gefunden.")
        return

    # Parse n_timesteps and file_model_type from first file
    name_only = str(model_files[0][2]).split("/")[-1].split("\\")[-1]
    parts = name_only.replace(".eqx", "").split("__")
    if len(parts) == 6:
        file_model_type = parts[0]
        n_timesteps = int(parts[4].replace("ts", ""))
    elif len(parts) == 5:
        file_model_type = parts[0]
        n_timesteps = int(parts[3].replace("ts", ""))
    else:
        print(f"Unbekanntes Format: {name_only}")
        return

    # Build model template
    key = jrandom.PRNGKey(0)
    if file_model_type == "gsm":
        model_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
    elif file_model_type == "simple_rnn":
        model_template = tm.build(key=key)
    elif file_model_type == "maxwell_nn":
        model_template = tm.build_maxwell_nn(
            key=key, E_infty=MATERIAL_PARAMS["E_infty"], E_val=MATERIAL_PARAMS["E"])
    else:
        print(f"Unbekannter Modelltyp: {file_model_type}")
        return

    n_om = len(test_omegas)
    n_A = len(test_As)

    # Use n_test_timesteps if provided, else n_timesteps from model file
    n_ts_test = n_test_timesteps if n_test_timesteps is not None else n_timesteps

    # Train info for annotations
    _TRAIN_INFO = {
        "omega_1": "ω={1}",
        "omega_2": "ω={1,2}",
        "omega_3": "ω={1,2,3}",
        "omega_4": "ω={1,2,3,4}",
        "amp_2":   "A={1,2}",
        "amp_3":   "A={1,2,3}",
        "amp_4":   "A={1,2,3,4}",
        "mixed_4": "(ω,A)∈{1,4}²",
        "mixed_2": "(ω,A)∈{1,2}²",
    }

    # Determine grid layout
    n_models = len(model_files)
    n_cols = min(n_models, 3)
    n_rows = (n_models + n_cols - 1) // n_cols

    # Use GridSpec with extra column for colorbar
    from matplotlib.gridspec import GridSpec
    cell_size = 5
    fig = plt.figure(figsize=(cell_size * n_cols + 2, cell_size * n_rows + 1))
    gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                  width_ratios=[1] * n_cols + [0.05], wspace=0.3, hspace=0.35)
    metric_name = "NRMSE" if normalize else "RMSE"
    noise_str = f", noise={noise_std_rel:.0%}" if noise_std_rel > 0 else ""
    ts_str = f", test_ts={n_ts_test}" if n_test_timesteps is not None else ""
    model_label = MODEL_LABELS.get(file_model_type, file_model_type.upper())
    fig.suptitle(f"{model_label} — {metric_name} Heatmaps ({test_type}) — {steps//1000}k steps{noise_str}{ts_str}", fontsize=13, y=0.98)

    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]
    cbar_ax = fig.add_subplot(gs[:, -1])

    # First pass: compute all RMSE values
    print("Berechne RMSE-Werte...")
    rmse_per_model = []
    for config, seed, filepath in model_files:
        model = storage.load_model(filepath, model_template)
        model = klax.finalize(model)

        rmse_grid = np.zeros((n_A, n_om))
        for i, A in enumerate(test_As):
            for j, omega in enumerate(test_omegas):
                eps, sig, dts = _generate_test_data(
                    n_ts_test, [omega], [A], test_type, noise_std_rel)
                sig_pred = jax.vmap(model)((eps, dts))
                rmse = float(np.sqrt(np.mean((np.array(sig_pred) - np.array(sig)) ** 2)))
                if normalize:
                    sig_std = float(np.std(np.array(sig)))
                    rmse = rmse / sig_std if sig_std > 1e-10 else rmse
                rmse_grid[i, j] = rmse

        rmse_per_model.append(rmse_grid)
        print(f"  {config} (seed {seed}): done")

    # Global color range
    all_vals = np.concatenate([r.ravel() for r in rmse_per_model])
    if log:
        log_floor = 1e-4
        norm = LogNorm(vmin=1e-4, vmax=4e-1)
    else:
        norm = None
        vmin = 0
        vmax = all_vals.max()

    # Second pass: plot
    im = None
    for idx, (config, seed, filepath) in enumerate(model_files):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]
        rmse_grid = rmse_per_model[idx]

        if log:
            plot_data = np.where(rmse_grid > 0, rmse_grid, log_floor)
            im = ax.imshow(plot_data, origin="lower", aspect="equal",
                           norm=norm, cmap="RdYlGn_r",
                           extent=[-0.5, n_om - 0.5, -0.5, n_A - 0.5])
        else:
            im = ax.imshow(rmse_grid, origin="lower", aspect="equal",
                           vmin=vmin, vmax=vmax, cmap="RdYlGn_r",
                           extent=[-0.5, n_om - 0.5, -0.5, n_A - 0.5])

        # Annotate cells with RMSE values (only if grid is small enough to read)
        if n_om <= 8 and n_A <= 8:
            thresh = norm(rmse_grid).data if log else rmse_grid / vmax if vmax > 0 else rmse_grid
            for i in range(n_A):
                for j in range(n_om):
                    val = rmse_grid[i, j]
                    t = thresh[i, j] if hasattr(thresh, '__getitem__') else 0.5
                    color = "white" if t > 0.6 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color=color)

        ax.set_xticks(range(n_om))
        ax.set_xticklabels(test_omegas, fontsize=6)
        ax.set_yticks(range(n_A))
        ax.set_yticklabels(test_As, fontsize=6)
        ax.set_xlabel("ω (test)")
        ax.set_ylabel("A (test)")

        train_info = _TRAIN_INFO.get(config, config)
        ax.set_title(f"{config}\nTrain: {train_info}, seed {seed}", fontsize=9)

    # Hide unused axes
    for idx in range(n_models, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row][col].axis("off")

    # Colorbar in its own axis
    label = f"{metric_name} (log)" if log else metric_name
    fig.colorbar(im, cax=cbar_ax, label=label)
    plt.show()


def plot_heatmaps_timestep_study(model_type, timestep_values=None, steps=100000,
                                  test_omegas=None, test_As=None,
                                  test_type="harmonic", log=True, normalize=True,
                                  noise_std_rel=0.0, search_dirs=None):
    """Plot RMSE heatmaps for a single model type across different timestep values.

    One heatmap per n_timesteps value, using best seed from BEST_SEEDS_TIMESTEP_STUDY.

    Args:
        model_type: "gsm", "maxwell_nn", or "simple_rnn"
        timestep_values: List of n_timesteps to show. Default: [50, 100, 200, 400]
        steps: Training steps filter (default: 100000)
        test_omegas: List of omega values for the grid
        test_As: List of A values for the grid
        test_type: "harmonic" or "relaxation"
        log: If True, use logarithmic color scale
        normalize: If True, use NRMSE
        noise_std_rel: Relative noise std on eps. Default: 0 (clean)
        search_dirs: Optional list of directories to search

    Examples:
        plot_heatmaps_timestep_study("gsm")
        plot_heatmaps_timestep_study("maxwell_nn", log=True)
        plot_heatmaps_timestep_study("simple_rnn", test_omegas=list(np.arange(0.5,6.5,0.5)))
    """
    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec

    if timestep_values is None:
        timestep_values = [50, 100, 200, 400]
    if search_dirs is None:
        search_dirs = ["artifacts/timestep_study"]
    if test_omegas is None:
        test_omegas = list(range(1, 21))
    if test_As is None:
        test_As = list(range(1, 21))

    ts_seeds = BEST_SEEDS_TIMESTEP_STUDY.get(model_type, {})
    if not ts_seeds:
        print(f"Keine best seeds für model_type='{model_type}' in BEST_SEEDS_TIMESTEP_STUDY")
        return

    # Collect model files
    model_files = []
    for n_ts in timestep_values:
        seed = ts_seeds.get(n_ts)
        if seed is None:
            print(f"Kein best seed für {model_type}, n_ts={n_ts}, überspringe...")
            continue
        pattern = f"{model_type}__mixed_4__seed_{seed}"
        f = find_latest(pattern, steps=steps, search_dirs=search_dirs, n_timesteps=n_ts)
        if f is not None:
            model_files.append((n_ts, seed, f))

    if not model_files:
        print("Keine Modelle gefunden.")
        return

    # Build model template
    key = jrandom.PRNGKey(0)
    if model_type == "gsm":
        model_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
    elif model_type == "simple_rnn":
        model_template = tm.build(key=key)
    elif model_type == "maxwell_nn":
        model_template = tm.build_maxwell_nn(
            key=key, E_infty=MATERIAL_PARAMS["E_infty"], E_val=MATERIAL_PARAMS["E"])
    else:
        print(f"Unbekannter Modelltyp: {model_type}")
        return

    n_om = len(test_omegas)
    n_A = len(test_As)

    # First pass: compute RMSE values
    print("Berechne RMSE-Werte...")
    rmse_per_model = []
    for n_ts, seed, filepath in model_files:
        model = storage.load_model(filepath, model_template)
        model = klax.finalize(model)

        rmse_grid = np.zeros((n_A, n_om))
        for i, A in enumerate(test_As):
            for j, omega in enumerate(test_omegas):
                eps, sig, dts = _generate_test_data(
                    n_ts, [omega], [A], test_type, noise_std_rel)
                sig_pred = jax.vmap(model)((eps, dts))
                rmse = float(np.sqrt(np.mean((np.array(sig_pred) - np.array(sig)) ** 2)))
                if normalize:
                    sig_std = float(np.std(np.array(sig)))
                    rmse = rmse / sig_std if sig_std > 1e-10 else rmse
                rmse_grid[i, j] = rmse

        rmse_per_model.append(rmse_grid)
        print(f"  {model_type} n_ts={n_ts} (seed {seed}): done")

    # Global color range
    metric_name = "NRMSE" if normalize else "RMSE"
    all_vals = np.concatenate([r.ravel() for r in rmse_per_model])
    if log:
        log_floor = 1e-4
        norm = LogNorm(vmin=1e-4, vmax=1e0)
    else:
        norm = None
        vmin = 0
        vmax = all_vals.max()

    # Layout
    n_models = len(model_files)
    n_cols = min(n_models, 4)
    n_rows = (n_models + n_cols - 1) // n_cols

    cell_size = 5
    fig = plt.figure(figsize=(cell_size * n_cols + 2, cell_size * n_rows + 1))
    gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                  width_ratios=[1] * n_cols + [0.05], wspace=0.3, hspace=0.35)

    model_label = MODEL_LABELS.get(model_type, model_type.upper())
    noise_str = f", noise={noise_std_rel:.0%}" if noise_std_rel > 0 else ""
    fig.suptitle(f"{model_label} — Timestep Study — {metric_name} ({test_type}){noise_str}",
                 fontsize=13, y=0.98)

    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]
    cbar_ax = fig.add_subplot(gs[:, -1])

    # Second pass: plot
    im = None
    for idx, (n_ts, seed, filepath) in enumerate(model_files):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]
        rmse_grid = rmse_per_model[idx]

        if log:
            plot_data = np.where(rmse_grid > 0, rmse_grid, log_floor)
            im = ax.imshow(plot_data, origin="lower", aspect="equal",
                           norm=norm, cmap="RdYlGn_r",
                           extent=[-0.5, n_om - 0.5, -0.5, n_A - 0.5])
        else:
            im = ax.imshow(rmse_grid, origin="lower", aspect="equal",
                           vmin=vmin, vmax=vmax, cmap="RdYlGn_r",
                           extent=[-0.5, n_om - 0.5, -0.5, n_A - 0.5])

        if n_om <= 8 and n_A <= 8:
            thresh = norm(rmse_grid).data if log else rmse_grid / vmax if vmax > 0 else rmse_grid
            for i in range(n_A):
                for j in range(n_om):
                    val = rmse_grid[i, j]
                    t = thresh[i, j] if hasattr(thresh, '__getitem__') else 0.5
                    color = "white" if t > 0.6 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color=color)

        ax.set_xticks(range(n_om))
        ax.set_xticklabels(test_omegas, fontsize=6)
        ax.set_yticks(range(n_A))
        ax.set_yticklabels(test_As, fontsize=6)
        ax.set_xlabel("ω (test)")
        ax.set_ylabel("A (test)")
        ax.set_title(f"n_timesteps={n_ts}\nseed {seed}", fontsize=9)

    # Hide unused axes
    for idx in range(n_models, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row][col].axis("off")

    label = f"{metric_name} (log)" if log else metric_name
    fig.colorbar(im, cax=cbar_ax, label=label)
    plt.show()


# =============================================================================
# Timestep Study: Aggregated Metrics (Compute + Save + Plot)
# =============================================================================

def compute_timestep_study_metrics(
    timestep_values=None,
    model_types=None,
    steps=100000,
    search_dirs=None,
    save_path="artifacts/timestep_study/metrics.json",
):
    """Compute aggregated NRMSE metrics for the timestep study and save to JSON.

    For each (model_type, n_timesteps) combination, computes NRMSE over two regions:
    - Interpolation: (omega, A) in [1, 4] x [1, 4] with 0.5 steps, excluding 4 training corners
      -> 7x7 - 4 = 45 points
    - Extrapolation: (omega, A) in [0.5, 6] x [0.5, 6] with 0.5 steps, minus the interpolation region
      -> 12x12 - 7x7 = 95 points

    Metrics computed per region: mean, median, geometric_mean, min, max of per-point NRMSE values.

    Args:
        timestep_values: List of n_timesteps. Default: [50, 100, 200, 400]
        model_types: List of model types. Default: ["gsm", "maxwell_nn", "simple_rnn"]
        steps: Training steps filter. Default: 100000
        search_dirs: Directories to search. Default: ["artifacts/timestep_study"]
        save_path: Path for the output JSON file

    Returns:
        dict with all computed metrics (also saved to JSON)
    """
    from scipy.stats import gmean

    if timestep_values is None:
        timestep_values = [50, 100, 200, 400]
    if model_types is None:
        model_types = ["gsm", "maxwell_nn", "simple_rnn"]
    if search_dirs is None:
        search_dirs = ["artifacts/timestep_study"]

    # Define grids
    interp_vals = [v / 2 for v in range(2, 9)]   # [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    extrap_vals = [v / 2 for v in range(1, 13)]   # [0.5, 1.0, ..., 5.5, 6.0]

    train_corners = {(1.0, 1.0), (1.0, 4.0), (4.0, 1.0), (4.0, 4.0)}
    interp_set = set((omega, A) for omega in interp_vals for A in interp_vals)

    interp_points = sorted(interp_set - train_corners)
    extrap_points = sorted(
        (omega, A) for omega in extrap_vals for A in extrap_vals
        if (omega, A) not in interp_set
    )

    print(f"Interpolation points: {len(interp_points)} (7x7 - 4 corners)")
    print(f"Extrapolation points: {len(extrap_points)} (12x12 - 7x7)")

    def _compute_nrmse_values(model, n_ts, points):
        """Compute per-point NRMSE for a list of (omega, A) pairs."""
        nrmse_vals = []
        for omega, A in points:
            eps, sig, dts = _generate_test_data(n_ts, [omega], [A], "harmonic", 0.0)
            sig_pred = jax.vmap(model)((eps, dts))
            rmse = float(np.sqrt(np.mean((np.array(sig_pred) - np.array(sig)) ** 2)))
            sig_std = float(np.std(np.array(sig)))
            nrmse = rmse / sig_std if sig_std > 1e-10 else rmse
            nrmse_vals.append(nrmse)
        return np.array(nrmse_vals)

    def _aggregate(vals):
        """Compute all aggregation metrics for an array of NRMSE values."""
        vals_pos = np.clip(vals, 1e-12, None)  # avoid log(0) for geometric mean
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "geometric_mean": float(gmean(vals_pos)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n_points": int(len(vals)),
        }

    # Build results
    results = {
        "description": "Aggregated NRMSE metrics for the timestep study",
        "normalization": "NRMSE = RMSE / std(sigma_true) per loadcase",
        "aggregation": "Metrics computed over per-point NRMSE values",
        "train_config": "mixed_4: (omega, A) in {1,4} x {1,4} (4 corners)",
        "interpolation_grid": {
            "omega_range": "[1.0, 4.0] step 0.5",
            "A_range": "[1.0, 4.0] step 0.5",
            "excluding": "4 training corners",
            "n_points": len(interp_points),
        },
        "extrapolation_grid": {
            "omega_range": "[0.5, 6.0] step 0.5",
            "A_range": "[0.5, 6.0] step 0.5",
            "excluding": "entire interpolation region [1,4]x[1,4]",
            "n_points": len(extrap_points),
        },
        "metrics": {},
    }

    for model_type in model_types:
        print(f"\n--- {model_type} ---")
        ts_seeds = BEST_SEEDS_TIMESTEP_STUDY.get(model_type, {})

        # Build model template once
        key = jrandom.PRNGKey(0)
        if model_type == "gsm":
            model_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
        elif model_type == "simple_rnn":
            model_template = tm.build(key=key)
        elif model_type == "maxwell_nn":
            model_template = tm.build_maxwell_nn(
                key=key, E_infty=MATERIAL_PARAMS["E_infty"], E_val=MATERIAL_PARAMS["E"])

        results["metrics"][model_type] = {}

        for n_ts in timestep_values:
            seed = ts_seeds.get(n_ts)
            if seed is None:
                print(f"  n_ts={n_ts}: kein best seed, überspringe...")
                continue

            pattern = f"{model_type}__mixed_4__seed_{seed}"
            f = find_latest(pattern, steps=steps, search_dirs=search_dirs, n_timesteps=n_ts)
            if f is None:
                continue

            model = storage.load_model(f, model_template)
            model = klax.finalize(model)

            interp_nrmse = _compute_nrmse_values(model, n_ts, interp_points)
            extrap_nrmse = _compute_nrmse_values(model, n_ts, extrap_points)

            results["metrics"][model_type][str(n_ts)] = {
                "seed": seed,
                "interpolation": _aggregate(interp_nrmse),
                "extrapolation": _aggregate(extrap_nrmse),
            }
            print(f"  n_ts={n_ts} (seed {seed}): "
                  f"interp geomean={gmean(np.clip(interp_nrmse, 1e-12, None)):.4e}, "
                  f"extrap geomean={gmean(np.clip(extrap_nrmse, 1e-12, None)):.4e}")

    # Save to JSON
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f_out:
        json.dump(results, f_out, indent=2)
    print(f"\nMetrics saved to: {save_path}")

    return results


def plot_timestep_study_metrics(
    metrics=None,
    metrics_path="artifacts/timestep_study/metrics.json",
    metric="geometric_mean",
    style="scatter",
):
    """Plot aggregated NRMSE metrics vs n_timesteps for the timestep study.

    Args:
        metrics: Dict from compute_timestep_study_metrics, or None to load from file
        metrics_path: Path to metrics JSON (used if metrics is None)
        metric: Which aggregation to plot. One of:
                "geometric_mean", "mean", "median", "min", "max"
        style: "scatter" (markers only, no lines), "bar" (grouped bar chart)

    Examples:
        plot_timestep_study_metrics()                                    # scatter (default)
        plot_timestep_study_metrics(style="bar")                         # grouped bars
        plot_timestep_study_metrics(metric="median", style="scatter")
    """
    if metrics is None:
        with open(metrics_path, "r") as f_in:
            metrics = json.load(f_in)

    model_types = list(metrics["metrics"].keys())
    colors = {
        "gsm": MODEL_COLORS.get("gsm", "#1f77b4"),
        "maxwell_nn": MODEL_COLORS.get("maxwell_nn", "#ff7f0e"),
        "simple_rnn": MODEL_COLORS.get("simple_rnn", "#2ca02c"),
    }

    metric_title = metric.replace("_", " ").title()

    if style == "bar":
        _plot_ts_metrics_bar(metrics, model_types, colors, metric, metric_title)
    else:
        _plot_ts_metrics_scatter(metrics, model_types, colors, metric, metric_title)


def _plot_ts_metrics_scatter(metrics, model_types, colors, metric, metric_title):
    """Scatter plot: markers only, no connecting lines."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for model_type in model_types:
        ts_data = metrics["metrics"][model_type]
        ts_values = sorted([int(k) for k in ts_data.keys()])

        interp_vals = [ts_data[str(ts)]["interpolation"][metric] for ts in ts_values]
        extrap_vals = [ts_data[str(ts)]["extrapolation"][metric] for ts in ts_values]

        label = MODEL_LABELS.get(model_type, model_type)
        color = colors.get(model_type, "gray")

        ax.scatter(ts_values, interp_vals, marker="o", s=80, color=color,
                   label=f"{label} (interpolation)", zorder=3)
        ax.scatter(ts_values, extrap_vals, marker="s", s=80, color=color,
                   facecolors="none", edgecolors=color, linewidths=2,
                   label=f"{label} (extrapolation)", zorder=3)

    ax.set_yscale("log")
    ax.set_xlabel("n_timesteps (training)")
    ax.set_ylabel(f"{metric_title} NRMSE")
    ax.set_title(f"Timestep Study — {metric_title} NRMSE\n"
                 f"Train: mixed_4 (corners), Interpolation vs Extrapolation")
    ax.set_xticks([50, 100, 200, 400])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_ts_metrics_bar(metrics, model_types, colors, metric, metric_title):
    """Grouped bar chart: interpolation + extrapolation side by side."""
    ts_data_first = metrics["metrics"][model_types[0]]
    ts_values = sorted([int(k) for k in ts_data_first.keys()])
    n_ts = len(ts_values)
    n_models = len(model_types)

    # Bar positions: group by n_timesteps, within each group one bar per (model, region)
    group_width = 0.8
    bar_width = group_width / (n_models * 2)
    x_base = np.arange(n_ts)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for m_idx, model_type in enumerate(model_types):
        ts_data = metrics["metrics"][model_type]
        interp_vals = [ts_data[str(ts)]["interpolation"][metric] for ts in ts_values]
        extrap_vals = [ts_data[str(ts)]["extrapolation"][metric] for ts in ts_values]

        label = MODEL_LABELS.get(model_type, model_type)
        color = colors.get(model_type, "gray")

        offset_interp = (m_idx * 2) * bar_width - group_width / 2 + bar_width / 2
        offset_extrap = (m_idx * 2 + 1) * bar_width - group_width / 2 + bar_width / 2

        ax.bar(x_base + offset_interp, interp_vals, bar_width * 0.9,
               color=color, alpha=0.9, label=f"{label} (interp.)")
        ax.bar(x_base + offset_extrap, extrap_vals, bar_width * 0.9,
               color=color, alpha=0.45, label=f"{label} (extrap.)",
               edgecolor=color, linewidth=1.2)

    ax.set_yscale("log")
    ax.set_xlabel("n_timesteps (training)")
    ax.set_ylabel(f"{metric_title} NRMSE")
    ax.set_title(f"Timestep Study — {metric_title} NRMSE\n"
                 f"Train: mixed_4 (corners), Interpolation vs Extrapolation")
    ax.set_xticks(x_base)
    ax.set_xticklabels([str(ts) for ts in ts_values])
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_training_loss(model_type=None, n_timesteps=None, seeds=None,
                       search_dirs=None, log=True):
    """Plot training loss curves from history JSON files.

    Flexible filtering: show all combos, or filter by model_type / n_timesteps / seeds.

    Args:
        model_type: Filter by model type (e.g. "gsm", "maxwell_nn", "simple_rnn"), or None for all
        n_timesteps: Filter by timestep value (e.g. 50, 100, 200, 400), or None for all
        seeds: List of seed indices to include, or None for all
        search_dirs: Directories to search. Default: ["artifacts/timestep_study"]
        log: If True, use log scale on y-axis. Default: True

    Examples:
        plot_training_loss()                                          # all
        plot_training_loss(model_type="gsm")                          # all GSM
        plot_training_loss(model_type="gsm", n_timesteps=200)         # GSM, 200 ts
        plot_training_loss(n_timesteps=400)                           # all models, 400 ts
        plot_training_loss(model_type="simple_rnn", seeds=[0, 1])     # RNN, seeds 0 & 1
    """
    from pathlib import Path

    if search_dirs is None:
        search_dirs = ["artifacts/timestep_study"]

    # Find all history JSONs
    all_jsons = []
    for d in search_dirs:
        p = Path(d)
        if p.exists():
            all_jsons.extend(p.glob("*_history.json"))

    if not all_jsons:
        print(f"Keine history JSONs gefunden in: {search_dirs}")
        return

    # Load and filter
    curves = []
    for jp in sorted(all_jsons):
        with open(jp, "r") as f_in:
            data = json.load(f_in)

        mt = data.get("model_type", "")
        nts = data.get("n_timesteps", 0)
        seed = data.get("seed", 0)
        losses = data.get("losses", [])

        if model_type is not None and mt != model_type:
            continue
        if n_timesteps is not None and nts != n_timesteps:
            continue
        if seeds is not None and seed not in seeds:
            continue
        if not losses:
            continue

        curves.append({
            "model_type": mt,
            "n_timesteps": nts,
            "seed": seed,
            "losses": losses,
            "final_loss": data.get("final_loss"),
        })

    if not curves:
        print("Keine passenden History-Dateien gefunden.")
        return

    print(f"Plotte {len(curves)} Trainingskurven")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Line styles per n_timesteps
    ts_linestyles = {50: "-", 100: "--", 200: "-.", 400: ":"}
    # Distinct seed colors (used when plotting a single model type)
    seed_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Detect if we're showing a single model type
    unique_models = set(c["model_type"] for c in curves)
    single_model = len(unique_models) == 1

    for c in curves:
        mt = c["model_type"]
        nts = c["n_timesteps"]
        seed = c["seed"]
        losses = c["losses"]

        if single_model:
            color = seed_colors[seed % len(seed_colors)]
        else:
            color = MODEL_COLORS.get(mt, "gray")
        ls = ts_linestyles.get(nts, "-")
        label = f"{MODEL_LABELS.get(mt, mt)} | ts={nts} | seed {seed}"

        # x-axis: steps (LOG_EVERY=500, so step_i = i * 500)
        steps_x = np.arange(1, len(losses) + 1) * 500
        ax.plot(steps_x, losses, color=color, linestyle=ls, alpha=0.85,
                linewidth=1.5, label=label)

    if log:
        ax.set_yscale("log")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Loss")

    # Build title from filters
    title_parts = ["Training Loss"]
    if model_type:
        title_parts.append(MODEL_LABELS.get(model_type, model_type))
    if n_timesteps:
        title_parts.append(f"ts={n_timesteps}")
    ax.set_title(" — ".join(title_parts))

    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_all_seeds(pattern, steps=None, seeds=None, test_loadcases=None, search_dirs=None,
                    noise_std_rel=0.0, n_timesteps=None):
    """Plot selected seeds for a config overlaid in one figure.

    Each seed gets a different color, ground truth is shown as black dashed line.
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    if test_loadcases is None:
        test_loadcases = [(1.0, 1.0)]

    As = [lc[0] for lc in test_loadcases]
    omegas = [lc[1] for lc in test_loadcases]

    # Collect filenames for requested seeds
    seed_files = []
    for seed in seeds:
        seed_pattern = f"{pattern}__seed_{seed}"
        f = find_latest(seed_pattern, steps=steps, search_dirs=search_dirs,
                        n_timesteps=n_timesteps)
        if f is not None:
            seed_files.append((seed, f))

    if not seed_files:
        print(f"Keine Modelle gefunden für Pattern '{pattern}'")
        return

    # Parse metadata from first file for title and model template
    name_only = str(seed_files[0][1]).split("/")[-1].split("\\")[-1]
    name_no_ext = name_only.replace(".eqx", "")
    parts = name_no_ext.split("__")

    if len(parts) == 6:
        model_type, experiment_name = parts[0], parts[1]
        train_steps = int(parts[3].replace("steps", ""))
        n_timesteps = int(parts[4].replace("ts", ""))
    elif len(parts) == 5:
        model_type, experiment_name = parts[0], parts[1]
        train_steps = int(parts[2].replace("steps", ""))
        n_timesteps = int(parts[3].replace("ts", ""))
    else:
        print(f"Unbekanntes Dateinamen-Format: {name_only}")
        return

    # Build model template
    key = jrandom.PRNGKey(0)
    if model_type == "simple_rnn":
        model_template = tm.build(key=key)
    elif model_type == "maxwell_nn":
        model_template = tm.build_maxwell_nn(
            key=key, E_infty=MATERIAL_PARAMS["E_infty"], E_val=MATERIAL_PARAMS["E"])
    elif model_type == "gsm":
        model_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
    else:
        print(f"Unbekannter Modelltyp: {model_type}")
        return

    # Generate test data
    eps_h, sig_h, dts_h = _generate_test_data(n_timesteps, omegas, As, "harmonic", noise_std_rel)
    eps_r, sig_r, dts_r = _generate_test_data(n_timesteps, omegas, As, "relaxation", noise_std_rel)

    n_pts = len(eps_h[0])
    ns = np.linspace(0, 2 * np.pi, n_pts)
    n_lc = len(test_loadcases)

    # Build title
    _TRAIN_INFO = {
        "omega_1": "Train: (A=1,ω=1)",
        "omega_2": "Train: (A=1,ω=1), (A=1,ω=2)",
        "omega_3": "Train: (A=1,ω=1..3)",
        "omega_4": "Train: (A=1,ω=1..4)",
        "amp_2":   "Train: (ω=1,A=1), (ω=1,A=2)",
        "amp_3":   "Train: (ω=1,A=1..3)",
        "amp_4":   "Train: (ω=1,A=1..4)",
        "mixed_4": "Train: (ω,A)∈{1,4}×{1,4}",
        "mixed_2": "Train: (ω,A)∈{1,2}×{1,2}",
    }
    train_info = _TRAIN_INFO.get(experiment_name, f"Train: {experiment_name}")
    noise_str = f" [noise={noise_std_rel:.0%}]" if noise_std_rel > 0 else ""
    base_title = f"{model_type.upper()} | {train_info} | {train_steps//1000}k steps | {len(seed_files)} seeds{noise_str}"

    # Seed colors (colormap)
    seed_cmap = plt.cm.tab10

    # --- Harmonic Plot ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{base_title} — Harmonic Test", fontsize=11)

    for i in range(n_lc):
        axs[0].plot(ns, sig_h[i], linestyle=":", color="black", linewidth=1.5,
                    label=f"GT: ω={omegas[i]}, A={As[i]}" if i == 0 or n_lc > 1 else None)
        axs[1].plot(eps_h[i], sig_h[i], linestyle=":", color="black", linewidth=1.5)

    for seed, filepath in seed_files:
        model = storage.load_model(filepath, model_template)
        model = klax.finalize(model)
        sig_pred = jax.vmap(model)((eps_h, dts_h))
        c = seed_cmap(seed)
        for i in range(n_lc):
            label = f"seed {seed}" if i == 0 else None
            axs[0].plot(ns, sig_pred[i], color=c, alpha=0.7, label=label)
            axs[1].plot(eps_h[i], sig_pred[i], color=c, alpha=0.7)

    axs[0].set_xlim([0, 2 * np.pi])
    axs[0].set_ylabel("stress $\\sigma$")
    axs[0].set_xlabel("time $t$")
    axs[0].legend(fontsize=8)
    axs[1].set_xlabel("strain $\\varepsilon$")
    axs[1].set_ylabel("stress $\\sigma$")
    fig.tight_layout()
    plt.show()

    # --- Relaxation Plot ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{base_title} — Relaxation Test", fontsize=11)

    for i in range(n_lc):
        axs[0].plot(ns, sig_r[i], linestyle=":", color="black", linewidth=1.5,
                    label=f"GT: ω={omegas[i]}, A={As[i]}" if i == 0 or n_lc > 1 else None)
        axs[1].plot(eps_r[i], sig_r[i], linestyle=":", color="black", linewidth=1.5)

    for seed, filepath in seed_files:
        model = storage.load_model(filepath, model_template)
        model = klax.finalize(model)
        sig_pred = jax.vmap(model)((eps_r, dts_r))
        c = seed_cmap(seed)
        for i in range(n_lc):
            label = f"seed {seed}" if i == 0 else None
            axs[0].plot(ns, sig_pred[i], color=c, alpha=0.7, label=label)
            axs[1].plot(eps_r[i], sig_pred[i], color=c, alpha=0.7)

    axs[0].set_xlim([0, 2 * np.pi])
    axs[0].set_ylabel("stress $\\sigma$")
    axs[0].set_xlabel("time $t$")
    axs[0].legend(fontsize=8)
    axs[1].set_xlabel("strain $\\varepsilon$")
    axs[1].set_ylabel("stress $\\sigma$")
    fig.tight_layout()
    plt.show()


def plot_saved_model(filename: str, test_loadcases=None, noise_std_rel=0.0):
    """Load and plot predictions for a saved model file.

    Args:
        filename: Path to the .eqx model file (relative or absolute)
        test_loadcases: List of (A, omega) tuples. Default: [(1,1), (1,2), (1,3)]
        noise_std_rel: Relative noise std on eps (e.g. 0.02 = 2%). Default: 0 (clean)
    """
    # 1. Metadaten aus Dateinamen extrahieren
    #    Altes Format (5 Teile): {model}__{experiment}__{steps}steps__{n}ts__{timestamp}.eqx
    #    Neues Format (6 Teile): {model}__{experiment}__seed_{i}__{steps}steps__{n}ts__{timestamp}.eqx
    try:
        name_only = str(filename).split("/")[-1].split("\\")[-1]
        name_no_ext = name_only.replace(".eqx", "")
        parts = name_no_ext.split("__")

        if len(parts) == 5:
            model_type = parts[0]
            experiment_name = parts[1]
            seed_str = ""
            train_steps = int(parts[2].replace("steps", ""))
            n_timesteps = int(parts[3].replace("ts", ""))
        elif len(parts) == 6:
            model_type = parts[0]
            experiment_name = parts[1]
            seed_str = parts[2]  # e.g. "seed_0"
            train_steps = int(parts[3].replace("steps", ""))
            n_timesteps = int(parts[4].replace("ts", ""))
        else:
            raise ValueError(f"Unbekanntes Format ({len(parts)} Teile): {name_only}")

        # Lesbaren Titel bauen
        # Trainings-Konfigurationen aus experiment_name ableiten
        _TRAIN_INFO = {
            "omega_1": "Train: (A=1,ω=1)",
            "omega_2": "Train: (A=1,ω=1), (A=1,ω=2)",
            "omega_3": "Train: (A=1,ω=1..3)",
            "omega_4": "Train: (A=1,ω=1..4)",
            "amp_2":   "Train: (ω=1,A=1), (ω=1,A=2)",
            "amp_3":   "Train: (ω=1,A=1..3)",
            "amp_4":   "Train: (ω=1,A=1..4)",
            "mixed_4": "Train: (ω,A)∈{1,4}×{1,4}",
            "mixed_2": "Train: (ω,A)∈{1,2}×{1,2}",
        }
        train_info = _TRAIN_INFO.get(experiment_name, f"Train: {experiment_name}")
        steps_info = f"{train_steps//1000}k steps"
        seed_info = f", {seed_str}" if seed_str else ""
        model_title = f"{model_type.upper()} | {train_info} | {steps_info}{seed_info}"

        print(f"Lade Modell: {model_title}")
    except (ValueError, IndexError) as e:
        print(f"Konnte Metadaten nicht aus Dateinamen lesen: {e}")
        return
    
    # 2. Modell-Template erstellen (für Equinox Load)
    key = jrandom.PRNGKey(0)
    if model_type == "simple_rnn":
        model_template = tm.build(key=key)
    elif model_type == "maxwell_nn":
        model_template = tm.build_maxwell_nn(
            key=key, 
            E_infty=MATERIAL_PARAMS["E_infty"], 
            E_val=MATERIAL_PARAMS["E"]
        )
    elif model_type == "gsm":
        model_template = tm.build_gsm(key=key, g=1.0/MATERIAL_PARAMS["eta"])
    else:
        print(f"Unbekannter Modelltyp: {model_type}")
        return

    # 3. Modell laden und finalisieren
    try:
        model = storage.load_model(filename, model_template)
        model = klax.finalize(model)
    except FileNotFoundError:
        print(f"Datei nicht gefunden: {filename}")
        return

    # 4. Testdaten generieren
    if test_loadcases is None:
        test_loadcases = [(1.0, 1.0), (1.0, 2.0), (1.0, 3.0)]
    As = [lc[0] for lc in test_loadcases]
    omegas = [lc[1] for lc in test_loadcases]
    
    noise_str = f" [noise={noise_std_rel:.0%}]" if noise_std_rel > 0 else ""

    # Harmonic Test
    print("Plotte Harmonic Test...")
    eps_h, sig_h, dts_h = _generate_test_data(n_timesteps, omegas, As, "harmonic", noise_std_rel)
    sig_pred_h = jax.vmap(model)((eps_h, dts_h))
    plot_model_pred(eps_h, sig_h, sig_pred_h, omegas, As,
                    title=f"{model_title} — Harmonic Test{noise_str}")

    # Relaxation Test
    print("Plotte Relaxation Test...")
    eps_r, sig_r, dts_r = _generate_test_data(n_timesteps, omegas, As, "relaxation", noise_std_rel)
    sig_pred_r = jax.vmap(model)((eps_r, dts_r))
    plot_model_pred(eps_r, sig_r, sig_pred_r, omegas, As,
                    title=f"{model_title} — Relaxation Test{noise_str}")
