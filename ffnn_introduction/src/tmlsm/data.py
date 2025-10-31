"""Data generation for task 1."""

import numpy as np
import klax
from matplotlib import pyplot as plt
import time
import jax.numpy as jnp
import jax


def bathtub():
    """Generate data for a bathtub function."""
    x = np.linspace(1, 10, 450)
    y = np.concatenate(
        [
            np.square(x[0:150] - 4) + 1,
            1 + 0.1 * np.sin(np.linspace(0, 3.14, 90)),
            np.ones(60),
            np.square(x[300:450] - 7) + 1,
        ]
    )

    x = x / 10.0
    y = y / 10.0

    x_cal = np.concatenate([x[0:240], x[330:420]])
    y_cal = np.concatenate([y[0:240], y[330:420]])

    return x, y, x_cal, y_cal

# --- The Ensemble Visualization Function ---
def plot_ensemble_results(ensemble_models, x, y, x_cal, y_cal, title):
    """
    Calculates ensemble statistics and plots the mean prediction with a
    shaded variance area.
    """
    # 1. Finalize all models in the ensemble
    finalized_models = [klax.finalize(m) for m in ensemble_models]

    # 2. Get predictions from every model
    # vmap each model and then apply it to the data `x`
    # This results in a list of prediction arrays
    predictions = [jax.vmap(m)(x) for m in finalized_models]
    
    # Stack predictions into a single (ensemble_size, num_data_points) array
    predictions_stack = jnp.stack(predictions)

    # 3. Calculate mean and standard deviation across the ensemble axis (axis=0)
    mean_preds = jnp.mean(predictions_stack, axis=0)
    std_preds = jnp.std(predictions_stack, axis=0)

    # 4. Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the true function and calibration data
    plt.scatter(x_cal[::10], y_cal[::10], c="green", label="Calibration Data", alpha=0.7, zorder=3)
    plt.plot(x, y, c="black", linestyle="--", label="Bathtub Function", zorder=4)

    # Plot the mean prediction of the ensemble
    plt.plot(x, mean_preds, label="Ensemble Mean", color="red", zorder=5)

    # Plot the shaded uncertainty region (+/- 2 standard deviations)
    plt.fill_between(
        x.flatten(),
        (mean_preds - 2 * std_preds).flatten(),
        (mean_preds + 2 * std_preds).flatten(),
        color="red",
        alpha=0.2,
        label="Ensemble Uncertainty (±2σ)",
        zorder=2
    )
    
    plt.title(f"Model Predictions for: {title}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.1, 1.1) # Set consistent y-axis limits for better comparison
    plt.show()

def plot_ensemble_histories(histories, title):
    """
    Plots the loss curves from multiple training histories on a single set of axes.

    Args:
        histories: A list of klax.HistoryCallback objects.
        title: The title for the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, history in enumerate(histories):
        # We pass the same 'ax' to every plot call.
        # We also customize the label for the legend.
        history.plot(
            ax=ax,
            loss_options={'label': f'Ensemble Run {i+1}', 'alpha': 0.8},
            val_loss_options={} # We don't have validation data, but this is here
        )

    ax.set_title(f"Training Loss History for: {title}")
    ax.set_yscale('log') # Loss is best viewed on a log scale
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()