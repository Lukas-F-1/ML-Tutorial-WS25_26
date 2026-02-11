"""Plotting utilities."""

from __future__ import annotations
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import numpy as np
import jax.numpy as jnp

if TYPE_CHECKING:
    from .experiments import ExperimentResult

from . import data as td
from . import models as tm
from . import storage
from . import evaluation as ev
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

def find_latest(pattern: str, steps=None, search_dirs=None) -> str:
    """Find the latest .eqx file matching a pattern.

    Args:
        pattern: Substring to match in filenames (e.g. "omega_3", "gsm__amp_4__seed_0")
        steps: Optional filter for training steps (e.g. 50000, 100000, 250000)
        search_dirs: List of directories to search. Default: all known artifact dirs

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

    if not matches:
        info = f"Pattern='{pattern}'"
        if steps is not None:
            info += f", steps={steps}"
        print(f"Keine .eqx Dateien gefunden mit {info}")
        print(f"Durchsuchte Ordner: {search_dirs}")
        return None

    # Sortiere nach Dateiname (Timestamp ist am Ende → lexikographisch sortierbar)
    matches.sort(key=lambda f: f.name)
    latest = matches[-1]
    print(f"Gefunden: {latest}")
    return str(latest)


def plot_latest(pattern: str, steps=None, test_loadcases=None, search_dirs=None,
                seeds=None, noise_std_rel=0.0):
    """Find the latest model matching a pattern and plot it.

    Args:
        pattern: Substring to match (e.g. "omega_3", "amp_4__seed_0", "maxwell_nn")
        steps: Optional filter for training steps (e.g. 50000, 150000, 250000)
        test_loadcases: List of (A, omega) tuples to test on. Default: [(1,1), (1,2), (1,3)]
        search_dirs: Optional list of directories to search
        seeds: List of seed indices to plot overlaid in one figure (e.g. [0,2,4])
               Use seeds=[0,1,2,3,4] for all 5 seeds.
        noise_std_rel: Relative noise std on eps (e.g. 0.02 = 2%). Default: 0 (clean)

    Examples:
        plot_latest("omega_3__seed_0")                              # single seed
        plot_latest("omega_3__seed_0", noise_std_rel=0.02)          # with 2% noise
        plot_latest("omega_3", seeds=[0,1,2,3,4])                   # all 5 seeds overlaid
    """
    if seeds is not None:
        _plot_all_seeds(pattern, steps=steps, seeds=seeds,
                        test_loadcases=test_loadcases, search_dirs=search_dirs,
                        noise_std_rel=noise_std_rel)
    else:
        filename = find_latest(pattern, steps=steps, search_dirs=search_dirs)
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


def _plot_all_seeds(pattern, steps=None, seeds=None, test_loadcases=None, search_dirs=None,
                    noise_std_rel=0.0):
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
        f = find_latest(seed_pattern, steps=steps, search_dirs=search_dirs)
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

def plot_best_gamma(
    configs=None,
    steps=250000,
    test_loadcases=None,
    search_dirs=None,
    noise_std_rel=0.0,
    n_test_timesteps=None,
):
    """
    Plot gamma(t) for the best GSM seed of each config vs analytical Maxwell gamma(t).

    Mirrors plot_best(...) but for internal variable gamma instead of stress sigma.
    Uses evaluation.simulate_model_batch to extract gamma trajectories.

    Args:
        configs: list like ["omega_4","amp_4","mixed_4"] (default: all BEST_SEEDS_GSM keys)
        steps: training steps filter, e.g. 250000
        test_loadcases: list of (A, omega) tuples, e.g. [(6,6)]
        search_dirs: optional override dirs; default GSM dirs
        noise_std_rel: relative noise on eps (0.0 for clean)
        n_test_timesteps: override number of timesteps in generated test trajectory
    """
    # --- choose best seeds + dirs (same logic as plot_best) ---
    best_seeds = BEST_SEEDS_GSM
    search_dirs = _get_search_dirs("gsm", search_dirs)

    if configs is None:
        configs = list(best_seeds.keys())
    if test_loadcases is None:
        test_loadcases = [(1.0, 1.0)]

    As = [lc[0] for lc in test_loadcases]
    omegas = [lc[1] for lc in test_loadcases]

    # --- find model files ---
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
        print("Keine GSM-Modelle gefunden.")
        return

    # --- determine timestep count from filename (same as plot_best) ---
    name_only = str(model_files[0][2]).split("/")[-1].split("\\")[-1]
    parts = name_only.replace(".eqx", "").split("__")
    if len(parts) == 6:
        n_timesteps = int(parts[4].replace("ts", ""))
    elif len(parts) == 5:
        n_timesteps = int(parts[3].replace("ts", ""))
    else:
        print(f"Unbekanntes Format: {name_only}")
        return

    n_ts_test = n_test_timesteps if n_test_timesteps is not None else n_timesteps

    # --- build templates for loading GSM + analytical Maxwell ---
    key = jrandom.PRNGKey(0)
    gsm_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
    maxwell_model = tm.build_maxwell(
        E_infty=MATERIAL_PARAMS["E_infty"],
        E_val=MATERIAL_PARAMS["E"],
        eta=MATERIAL_PARAMS["eta"],
    )

    # --- generate test data (eps, dts) ---
    eps_h, sig_h, dts_h = _generate_test_data(
        n_ts_test, omegas, As, test_type="harmonic", noise_std_rel=noise_std_rel
    )

    # --- simulate Maxwell to get "true" gamma ---
    gamma_max, _ = ev.simulate_model_batch(maxwell_model, eps_h, dts_h)  # gamma: (N,T+1)

    # time axis (keep consistent with your other plots)
    T = eps_h.shape[1]
    t = np.linspace(0, 2 * np.pi, T + 1)

    # --- plot: one figure per loadcase (recommended) ---
    cmap = plt.cm.tab10
    noise_str = f", noise={noise_std_rel:.0%}" if noise_std_rel > 0 else ""
    ts_str = f", test_ts={n_ts_test}" if n_test_timesteps is not None else ""
    tc_str = ", ".join([f"(A={a},ω={w})" for a, w in test_loadcases])

    for lc_idx, (A, omega) in enumerate(test_loadcases):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_title(
            f"GSM γ(t) vs Maxwell γ(t) — Test {tc_str}{noise_str}{ts_str}\n"
            f"Showing loadcase A={A}, ω={omega}"
        )

        # ground truth gamma
        ax.plot(t, gamma_max[lc_idx], linestyle=":", color="black", linewidth=2, label="Maxwell γ (GT)")

        # each GSM config
        for idx, (config, seed, filepath) in enumerate(model_files):
            model = storage.load_model(filepath, gsm_template)
            model = klax.finalize(model)
            gamma_gsm, _ = ev.simulate_model_batch(model, eps_h, dts_h)
            c = cmap(idx % 10)
            ax.plot(t, gamma_gsm[lc_idx], color=c, alpha=0.85, linewidth=1.8, label=f"{config} (s{seed})")

        ax.set_xlabel("time t")
        ax.set_ylabel("internal variable γ")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        plt.show()

def plot_best_teacher_forced(
    configs=None,
    steps=250000,
    test_loadcases=None,
    search_dirs=None,
    noise_std_rel=0.0,
    n_test_timesteps=None,
    plot_option_c: bool = True,
):

    """
    Option B diagnostic (teacher forcing) for GSM:

    Compare, for each best-seed GSM model:
      - closed-loop GSM sigma(t)
      - teacher-forced GSM sigma_TF(t) using gamma_true(t) from Maxwell
      - ground-truth Maxwell sigma(t) (from data generator)

    Works without retraining and without requiring a separate loader in user code.
    """

    # ----------- Select models (same scheme as plot_best) -----------
    model_type = "gsm"
    best_seeds = _get_best_seeds(model_type)
    search_dirs = _get_search_dirs(model_type, search_dirs)

    if configs is None:
        configs = list(best_seeds.keys())
    if test_loadcases is None:
        test_loadcases = [(1.0, 1.0)]

    As = [lc[0] for lc in test_loadcases]
    omegas = [lc[1] for lc in test_loadcases]

    model_files = []
    for config in configs:
        seed = best_seeds.get(config)
        if seed is None:
            print(f"Kein best seed definiert für '{config}', überspringe.")
            continue
        pattern = f"{config}__seed_{seed}"
        f = find_latest(pattern, steps=steps, search_dirs=search_dirs)
        if f is not None:
            model_files.append((config, seed, f))

    if not model_files:
        print("Keine Modelle gefunden.")
        return

    # ----------- Parse n_timesteps from first file (same as plot_best) -----------
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

    if file_model_type != "gsm":
        print(f"plot_best_teacher_forced ist nur für GSM gedacht, Datei-Typ war: {file_model_type}")
        return

    # ----------- Build templates (GSM + Maxwell) -----------
    key = jrandom.PRNGKey(0)
    gsm_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
    maxwell_model = tm.build_maxwell(
        E_infty=MATERIAL_PARAMS["E_infty"],
        E_val=MATERIAL_PARAMS["E"],
        eta=MATERIAL_PARAMS["eta"],
    )

    # ----------- Generate test data (harmonic only for this diagnostic) -----------
    n_ts_test = n_test_timesteps if n_test_timesteps is not None else n_timesteps
    eps_h, sig_h, dts_h = _generate_test_data(n_ts_test, omegas, As, "harmonic", noise_std_rel)

    # time axis
    n_pts = len(eps_h[0])
    ts = np.linspace(0, 2 * np.pi, n_pts)

    # ----------- Get gamma_true(t) from Maxwell (evaluation only) -----------
    gamma_max, _ = ev.simulate_model_batch(maxwell_model, eps_h, dts_h)  # (N, T+1)

    # ----------- Teacher-forced sigma helper (no need to modify evaluation.py) -----------
    def teacher_forced_sigma(gsm_model, eps_batch, gamma_true_batch):
        """
        sigma_TF(t) = d/d_eps e_theta(eps(t), gamma_true(t))
        eps_batch: (N,T)
        gamma_true_batch: (N,T+1) or (N,T)
        returns: (N,T)
        """
        cell = gsm_model.cell
        if not hasattr(cell, "_energy"):
            raise ValueError("Given model does not look like GSM (missing cell._energy).")

        # align gamma length to T
        if gamma_true_batch.shape[1] == eps_batch.shape[1] + 1:
            gamma_T = gamma_true_batch[:, : eps_batch.shape[1]]
        else:
            gamma_T = gamma_true_batch

        de_deps = jax.grad(cell._energy, argnums=0)

        # vmap over time, then batch
        def one_case(eps_1, gam_1):
            return jax.vmap(de_deps)(eps_1, gam_1)

        sig_tf = jax.vmap(one_case)(jnp.asarray(eps_batch), jnp.asarray(gamma_T))
        return np.array(sig_tf)
    
    def teacher_forced_dsig_dgamma(gsm_model, eps_batch, gamma_forced_batch):
        """
        Compute dσ/dγ along a provided gamma trajectory:
            σ = ∂e/∂ε
            dσ/dγ = ∂/∂γ (∂e/∂ε) = ∂²e/(∂γ∂ε)

        eps_batch: (N,T)
        gamma_forced_batch: (N,T+1) or (N,T)
        returns: (N,T)
        """
        cell = gsm_model.cell
        if not hasattr(cell, "_energy"):
            raise ValueError("Given model does not look like GSM (missing cell._energy).")

        if gamma_forced_batch.shape[1] == eps_batch.shape[1] + 1:
            gamma_T = gamma_forced_batch[:, : eps_batch.shape[1]]
        else:
            gamma_T = gamma_forced_batch

        # σ(eps,gamma) = ∂e/∂eps
        de_deps = jax.grad(cell._energy, argnums=0)
        # dσ/dγ = ∂/∂γ (∂e/∂eps)
        dsig_dgamma = jax.grad(de_deps, argnums=1)

        def one_case(eps_1, gam_1):
            return jax.vmap(dsig_dgamma)(eps_1, gam_1)

        out = jax.vmap(one_case)(jnp.asarray(eps_batch), jnp.asarray(gamma_T))
        return np.array(out)


    # ----------- Plotting -----------
    n_lc = len(test_loadcases)
    cmap = plt.cm.tab10

    tc_str = ", ".join([f"(A={a},ω={w})" for a, w in test_loadcases])
    noise_str = f", noise={noise_std_rel:.0%}" if noise_std_rel > 0 else ""
    ts_str = f", test_ts={n_ts_test}" if n_test_timesteps is not None else ""

    E_true = MATERIAL_PARAMS["E"]
    dsig_dg_true = -float(E_true)


    # One figure per loadcase (keeps legend readable)
    for i_lc in range(n_lc):
        if plot_option_c:
            fig, axs = plt.subplots(2, 3, figsize=(16, 9))
            ax_t = axs[0, 0]
            ax_loop = axs[0, 1]
            ax_sens = axs[0, 2]

            # Bottom row (Option C panels)
            ax_sens_vs_gamma = axs[1, 0]
            ax_state_colored = axs[1, 1]
            ax_extra         = axs[1, 2]
        else:
            fig, axs = plt.subplots(1, 3, figsize=(16, 5))
            ax_t, ax_loop, ax_sens = axs
            # No bottom row in this case
            ax_sens_vs_gamma = None
            ax_state_colored = None
            ax_extra = None

        fig.suptitle(
            f"GSM Option B (Teacher forcing) — Harmonic — Test: {tc_str}{noise_str}{ts_str}\n"
            f"Showing loadcase A={As[i_lc]}, ω={omegas[i_lc]}",
            fontsize=11,
        )

        # Ground truth (from generator)
        ax_t.plot(ts, sig_h[i_lc], linestyle=":", color="black", linewidth=2, label="Ground Truth")
        ax_loop.plot(eps_h[i_lc], sig_h[i_lc], linestyle=":", color="black", linewidth=2)

        # Each best model: plot closed-loop (solid) + teacher-forced (dashed) in same color
        for idx, (config, seed, filepath) in enumerate(model_files):
            model = storage.load_model(filepath, gsm_template)
            model = klax.finalize(model)

            # closed-loop sigma
            sig_closed = np.array(jax.vmap(model)((eps_h, dts_h)))[i_lc]

            # teacher-forced sigma
            sig_tf = teacher_forced_sigma(model, eps_h, gamma_max)[i_lc]

            # gamma sensitivity of gsm energy
            dsig_dg_tf = teacher_forced_dsig_dgamma(model, eps_h, gamma_max)[i_lc]


            # sanity check
            gamma_gsm, _ = ev.simulate_model_batch(model, eps_h, dts_h)
            gmax = gamma_max[i_lc, :eps_h.shape[1]]
            ggsm = gamma_gsm[i_lc, :eps_h.shape[1]]
            print("max|γ_gsm-γ_max| =", np.max(np.abs(ggsm - gmax)))
            print("max|σ_closed-σ_TF| =", np.max(np.abs(sig_closed - sig_tf)))


            c = cmap(idx % 10)
            label_closed = f"{config} (s{seed}) closed" if i_lc == 0 else None
            label_tf = f"{config} (s{seed}) TF" if i_lc == 0 else None

            ax_t.plot(ts, sig_closed, color=c, alpha=0.85, linewidth=1.8, label=label_closed)
            ax_t.plot(ts, sig_tf, color="red", alpha=0.85, linewidth=1.8, linestyle="--", label=label_tf)

            ax_loop.plot(eps_h[i_lc], sig_closed, color=c, alpha=0.85, linewidth=1.8)
            ax_loop.plot(eps_h[i_lc], sig_tf, color=c, alpha=0.85, linewidth=1.8, linestyle="--")

            ax_sens.plot(ts, dsig_dg_tf, color=c, alpha=0.85, linewidth=1.4)

            if plot_option_c:
                # align teacher-forced gamma to length T
                gamma_tf_T = gamma_max[i_lc, :eps_h.shape[1]]
                eps_T = eps_h[i_lc]

                # (Row 2, Col 1): dσ/dγ vs γ, colored by ε
                sc1 = ax_sens_vs_gamma.scatter(
                    gamma_tf_T, dsig_dg_tf, c=eps_T, s=12, alpha=0.8
                )

                # (Row 2, Col 2): ε-γ state trajectory colored by dσ/dγ
                sc2 = ax_state_colored.scatter(
                    eps_T, gamma_tf_T, c=dsig_dg_tf, s=12, alpha=0.8
                )

                # also draw the path lightly to show loop ordering
                ax_state_colored.plot(eps_T, gamma_tf_T, linewidth=0.8, alpha=0.3)



        ax_t.set_xlim([0, 2 * np.pi])
        ax_t.set_xlabel("time $t$")
        ax_t.set_ylabel("stress $\\sigma$")
        ax_t.legend(fontsize=7, loc="best")

        ax_loop.set_xlabel("strain $\\varepsilon$")
        ax_loop.set_ylabel("stress $\\sigma$")

        ax_sens.set_xlabel("time $t$")
        ax_sens.set_ylabel(r"sensitivity $\partial\sigma/\partial\gamma$")
        ax_sens.set_title(r"$\partial\sigma/\partial\gamma$ (teacher-forced $\gamma_{max}$)")
        ax_sens.grid(alpha=0.25)
        # Existing (top row) sensitivity plot formatting
        ax_sens.axhline(dsig_dg_true, color="black", linestyle=":", linewidth=1.5, label="Maxwell: -E")
        ax_sens.legend(fontsize=7, loc="best")

        if plot_option_c:
            # Bottom-left: dσ/dγ vs γ
            ax_sens_vs_gamma.axhline(dsig_dg_true, color="black", linestyle=":", linewidth=1.5)
            ax_sens_vs_gamma.set_xlabel("internal variable γ (teacher-forced)")
            ax_sens_vs_gamma.set_ylabel(r"sensitivity $\partial\sigma/\partial\gamma$")
            ax_sens_vs_gamma.set_title(r"$\partial\sigma/\partial\gamma$ vs γ (colored by ε)")
            ax_sens_vs_gamma.grid(alpha=0.25)
            cbar1 = fig.colorbar(sc1, ax=ax_sens_vs_gamma)
            cbar1.set_label("strain ε")

            # Bottom-middle: ε–γ trajectory colored by sensitivity
            ax_state_colored.set_xlabel("strain ε")
            ax_state_colored.set_ylabel("internal variable γ (teacher-forced)")
            ax_state_colored.set_title("State trajectory (ε,γ) colored by ∂σ/∂γ")
            ax_state_colored.grid(alpha=0.25)
            cbar2 = fig.colorbar(sc2, ax=ax_state_colored)
            cbar2.set_label(r"$\partial\sigma/\partial\gamma$")

            # Bottom-right: leave empty or use for Δσ(t) later
            ax_extra.axis("off")




        fig.tight_layout()
        plt.show()

def plot_best_state_curvature(
    configs=None,
    steps=250000,
    search_dirs=None,
    noise_std_rel=0.0,
    n_test_timesteps=100,
    test_loadcases=None,
    eps_range=None,
    gamma_range=None,
    n_grid=120,
    show_training_states=True,
    show_test_states=True,
):
    """
    Plot GSM energy curvature fields over state space (ε,γ) for best-seed models:
      - k_eps_eps(ε,γ) = ∂²e/∂ε²   (effective stiffness -> saturation if ~0)
      - k_eps_gam(ε,γ) = ∂²e/(∂γ∂ε) = ∂σ/∂γ  (coupling -> decoupling if ~0)

    Overlays:
      - training trajectories (states visited by training loadcases for that config)
      - selected test trajectories (user-provided A,ω pairs)

    Parameters follow the same philosophy as plot_best(...):
      configs: list of config names (e.g. ["mixed_4"]) or None for all best seeds
      steps: train steps filter
      test_loadcases: list of (A, omega) tuples to overlay as OOD/ID examples
      eps_range/gamma_range: (min,max) for heatmap; if None, derived from overlay trajectories
      n_grid: resolution per axis
    """

    import numpy as np
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import matplotlib.pyplot as plt

    best_seeds = _get_best_seeds("gsm")
    search_dirs = _get_search_dirs("gsm", search_dirs)

    if configs is None:
        configs = list(best_seeds.keys())
    if test_loadcases is None:
        test_loadcases = [(1.5, 1.5), (6.0, 6.0)]  # sensible default examples

    # Helper: training loadcases per experiment name (matches your plots.py TRAIN_INFO meaning)
    def _train_loadcases_for_config(name: str):
        # NOTE: This mapping matches how you've described/used omega_k, amp_k, mixed_k in your slides/plots.
        if name.startswith("omega_"):
            k = int(name.split("_")[1])
            return [(1.0, float(w)) for w in range(1, k + 1)]
        if name.startswith("amp_"):
            k = int(name.split("_")[1])
            return [(float(a), 1.0) for a in range(1, k + 1)]
        if name == "mixed_4":
            vals = [1.0, 2.0, 3.0, 4.0]
            return [(A, w) for A in vals for w in vals]
        if name == "mixed_2":
            vals = [1.0, 2.0]
            return [(A, w) for A in vals for w in vals]
        # fallback: assume trained on (1,1)
        return [(1.0, 1.0)]

    # Collect model files
    model_files = []
    for config in configs:
        seed = best_seeds.get(config)
        if seed is None:
            print(f"Kein best seed definiert für '{config}', überspringe.")
            continue
        pattern = f"{config}__seed_{seed}"
        f = find_latest(pattern, steps=steps, search_dirs=search_dirs)
        if f is not None:
            model_files.append((config, seed, f))

    if not model_files:
        print("Keine Modelle gefunden.")
        return

    # Build GSM template for loading
    key = jrandom.PRNGKey(0)
    gsm_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])

    # Local helper to get (eps,gamma) trajectories from your analytical generator
    def _states_for_loadcases(loadcases):
        As = [lc[0] for lc in loadcases]
        omegas = [lc[1] for lc in loadcases]
        if noise_std_rel > 0:
            eps, gam, sig, dts = td.generate_data_harmonic_noisy_eps(
                MATERIAL_PARAMS["E_infty"], MATERIAL_PARAMS["E"], MATERIAL_PARAMS["eta"],
                n_test_timesteps, omegas, As,
                noise_std_rel=noise_std_rel, seed=0, recompute_eps_dot_from_noisy=False
            )
        else:
            eps, gam, sig, dts = td.generate_data_harmonic(
                MATERIAL_PARAMS["E_infty"], MATERIAL_PARAMS["E"], MATERIAL_PARAMS["eta"],
                n_test_timesteps, omegas, As
            )
        # eps,gam are (N,T)
        return np.asarray(eps), np.asarray(gam)

    # Compute default plotting ranges from overlays if not given
    def _infer_ranges(eps_train, gam_train, eps_test, gam_test):
        eps_all = []
        gam_all = []
        if eps_train is not None:
            eps_all.append(eps_train.reshape(-1))
            gam_all.append(gam_train.reshape(-1))
        if eps_test is not None:
            eps_all.append(eps_test.reshape(-1))
            gam_all.append(gam_test.reshape(-1))
        eps_all = np.concatenate(eps_all) if eps_all else np.array([-1, 1], dtype=float)
        gam_all = np.concatenate(gam_all) if gam_all else np.array([-1, 1], dtype=float)
        # small margins
        emn, emx = float(np.min(eps_all)), float(np.max(eps_all))
        gmn, gmx = float(np.min(gam_all)), float(np.max(gam_all))
        pad_e = 0.08 * (emx - emn + 1e-9)
        pad_g = 0.08 * (gmx - gmn + 1e-9)
        return (emn - pad_e, emx + pad_e), (gmn - pad_g, gmx + pad_g)

    # --- Main: one figure per config (2 heatmaps side-by-side) ---
    for config, seed, filepath in model_files:
        model = storage.load_model(filepath, gsm_template)
        model = klax.finalize(model)

        # training + test trajectories (state overlays)
        train_lcs = _train_loadcases_for_config(config)
        eps_tr, gam_tr = _states_for_loadcases(train_lcs) if show_training_states else (None, None)
        eps_te, gam_te = _states_for_loadcases(test_loadcases) if show_test_states else (None, None)

        # heatmap ranges
        if eps_range is None or gamma_range is None:
            (e0, e1), (g0, g1) = _infer_ranges(eps_tr, gam_tr, eps_te, gam_te)
        else:
            e0, e1 = eps_range
            g0, g1 = gamma_range

        # Build state grid
        eps_lin = jnp.linspace(e0, e1, n_grid)
        gam_lin = jnp.linspace(g0, g1, n_grid)
        EE, GG = jnp.meshgrid(eps_lin, gam_lin, indexing="xy")  # (n_grid,n_grid)

        # Access GSM energy function e(ε,γ)
        cell = model.cell
        if not hasattr(cell, "_energy"):
            raise ValueError("Model does not look like GSM (missing model.cell._energy).")

        def e_fun(eps_s, gam_s):
            return cell._energy(eps_s, gam_s)

        # Curvatures:
        # k_eps_eps = ∂²e/∂ε²
        # k_eps_gam = ∂²e/(∂γ∂ε) = ∂/∂γ(∂e/∂ε)
        de_deps = jax.grad(e_fun, argnums=0)
        k_eps_eps_fun = jax.grad(de_deps, argnums=0)
        k_eps_gam_fun = jax.grad(de_deps, argnums=1)

        # Vectorize over grid points
        pts = jnp.stack([EE.reshape(-1), GG.reshape(-1)], axis=1)  # (n_grid^2, 2)

        @jax.jit
        def eval_fields(pts_):
            eps_v = pts_[:, 0]
            gam_v = pts_[:, 1]
            k11 = jax.vmap(k_eps_eps_fun)(eps_v, gam_v)
            k12 = jax.vmap(k_eps_gam_fun)(eps_v, gam_v)
            return k11, k12

        k11_flat, k12_flat = eval_fields(pts)
        K11 = np.array(k11_flat).reshape(n_grid, n_grid)
        K12 = np.array(k12_flat).reshape(n_grid, n_grid)

        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"GSM curvature fields — {config} (s{seed}) — {steps//1000}k steps\n"
            f"Heatmaps over state space (ε,γ), with trajectory overlays",
            fontsize=11
        )

        # Heatmap 1: stiffness in ε
        im0 = axs[0].imshow(
            K11,
            origin="lower",
            extent=[e0, e1, g0, g1],
            aspect="auto"
        )
        axs[0].set_title(r"$k_{\varepsilon\varepsilon}=\partial^2 e/\partial\varepsilon^2$  (stiffness)")
        axs[0].set_xlabel("strain ε")
        axs[0].set_ylabel("internal variable γ")
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        # Heatmap 2: coupling
        im1 = axs[1].imshow(
            K12,
            origin="lower",
            extent=[e0, e1, g0, g1],
            aspect="auto"
        )
        axs[1].set_title(r"$k_{\varepsilon\gamma}=\partial^2 e/(\partial\gamma\,\partial\varepsilon)=\partial\sigma/\partial\gamma$")
        axs[1].set_xlabel("strain ε")
        axs[1].set_ylabel("internal variable γ")
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        # Overlay training trajectories (light gray)
        if show_training_states and eps_tr is not None:
            axs[0].scatter(eps_tr.reshape(-1), gam_tr.reshape(-1), s=3, alpha=0.12, color="white", edgecolors="none", label="train states")
            axs[1].scatter(eps_tr.reshape(-1), gam_tr.reshape(-1), s=3, alpha=0.12, color="white", edgecolors="none", label="train states")

        # Overlay selected test trajectories (colored)
        if show_test_states and eps_te is not None:
            cmap = plt.cm.tab10
            for i, (A, w) in enumerate(test_loadcases):
                c = cmap(i % 10)
                axs[0].plot(eps_te[i], gam_te[i], color=c, linewidth=1.2, alpha=0.9, label=f"test (A={A},ω={w})")
                axs[1].plot(eps_te[i], gam_te[i], color=c, linewidth=1.2, alpha=0.9, label=f"test (A={A},ω={w})")

        for ax in axs:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, loc="best")

        fig.tight_layout()
        plt.show()
