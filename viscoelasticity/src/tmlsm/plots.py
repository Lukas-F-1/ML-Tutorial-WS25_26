"""Plotting utilities."""

from __future__ import annotations
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
# Single Model Plotting
# =============================================================================

def find_latest(pattern: str, steps=None, search_dirs=None) -> str:
    """Find the latest .eqx file matching a pattern.

    Args:
        pattern: Substring to match in filenames (e.g. "omega_3", "gsm__amp_4__seed_0")
        steps: Optional filter for training steps (e.g. 50000, 100000, 250000)
        search_dirs: List of directories to search. Default: ["artifacts", "artifacts/gsm_experiments"]

    Returns:
        Path to the latest matching .eqx file (sorted by timestamp in filename)
    """
    from pathlib import Path

    if search_dirs is None:
        search_dirs = ["artifacts", "artifacts/gsm_experiments"]

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


def plot_latest(pattern: str, steps=None, test_loadcases=None, search_dirs=None):
    """Find the latest model matching a pattern and plot it.

    Args:
        pattern: Substring to match (e.g. "omega_3", "amp_4__seed_0", "maxwell_nn")
        steps: Optional filter for training steps (e.g. 50000, 150000, 250000)
        test_loadcases: List of (A, omega) tuples to test on. Default: [(1,1), (1,2), (1,3)]
        search_dirs: Optional list of directories to search

    Examples:
        plot_latest("omega_3")                    # latest omega_3 (any seed, 250k)
        plot_latest("omega_3__seed_0", steps=50000)  # omega_3 seed 0 at 50k checkpoint
        plot_latest("omega_3", test_loadcases=[(2,4), (3,1)])  # custom test cases
    """
    filename = find_latest(pattern, steps=steps, search_dirs=search_dirs)
    if filename is not None:
        plot_saved_model(filename, test_loadcases=test_loadcases)


def plot_saved_model(filename: str, test_loadcases=None):
    """Load and plot predictions for a saved model file.

    Args:
        filename: Path to the .eqx model file (relative or absolute)
        test_loadcases: List of (A, omega) tuples. Default: [(1,1), (1,2), (1,3)]
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
    
    # Harmonic Test
    print("Plotte Harmonic Test...")
    eps_h, _, sig_h, dts_h = td.generate_data_harmonic(
        MATERIAL_PARAMS["E_infty"],
        MATERIAL_PARAMS["E"],
        MATERIAL_PARAMS["eta"],
        n_timesteps,
        omegas,
        As
    )
    # Vorhersage (vmap über Batch-Dimension)
    sig_pred_h = jax.vmap(model)((eps_h, dts_h))
    plot_model_pred(eps_h, sig_h, sig_pred_h, omegas, As,
                    title=f"{model_title} — Harmonic Test")

    # Relaxation Test
    print("Plotte Relaxation Test...")
    eps_r, _, sig_r, dts_r = td.generate_data_relaxation(
        MATERIAL_PARAMS["E_infty"],
        MATERIAL_PARAMS["E"],
        MATERIAL_PARAMS["eta"],
        n_timesteps,
        omegas,
        As
    )
    # Vorhersage
    sig_pred_r = jax.vmap(model)((eps_r, dts_r))
    plot_model_pred(eps_r, sig_r, sig_pred_r, omegas, As,
                    title=f"{model_title} — Relaxation Test")
