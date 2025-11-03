import numpy as np
import klax
from matplotlib import pyplot as plt
import jax.numpy as jnp
import jax



# ----- Default bathtub data generation function (unchanged) ----- #
"""Tasks: 1.1 and 1.2"""


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



# ----- 1D plotting function adjusted to ensemble method explained in main.jpynb ----- #
"""Tasks: 1.1 and 1.2"""


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



# ----- Training history plotting function adjusted to ensemble method explained in main.jpynb ----- #
"""Tasks: 1.1, 1.2, 2.2, 2.3, 2.4"""


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



# ----- Simple 1D plotting function for sobolev stress strain model in task 2.4 ----- #
"""Tasks: 2.1, 2.2, 2.3"""


def generate_2d_grid():
    """
    Generates a 2D grid of (x, y) points.
    
    Generates 20 equidistant points from -4 to 4 for both x and y,
    and returns the meshgrid, the flattened (x, y) pairs for the NN,
    and the x and y linspaces.
    """
    # 1. Create the 20 equidistant points
    x_lin = jnp.linspace(-4, 4, 20)
    y_lin = jnp.linspace(-4, 4, 20)
    
    # 2. Create the (20, 20) meshgrid for 3D plotting
    xx, yy = jnp.meshgrid(x_lin, y_lin)
    
    # 3. Create the (400, 2) input data for the NN
    # We flatten the grids and stack them
    x_data = jnp.stack([xx.flatten(), yy.flatten()], axis=1)
    
    return x_data, xx, yy, x_lin, y_lin



# ----- Data generation function for f1 ----- #
"""Tasks: 2.1, 2.2"""


def f1():
    """Generates data for f1(x) = x^2 - y^2"""
    # 1. Generate the full (20x20) grid for plotting
    x_data_full, xx, yy, _, _ = generate_2d_grid()
    f1_grid = xx**2 - yy**2
    f1_data_full = f1_grid.reshape(-1, 1)
    
    # 2. Format calibration data for the NN
    x_cal = x_data_full
    y_cal = f1_data_full

    return x_data_full, f1_data_full, xx, yy, f1_grid, x_cal, y_cal



# ----- Data generation function for f2 ----- #
"""Tasks: 2.1, 2.2, 2.3"""


def f2():
    """Generates data for f2(x) = x^2 + 0.5y^2"""
    # 1. Generate the full (20x20) grid for plotting
    x_data_full, xx, yy, _, _ = generate_2d_grid()
    f2_grid = xx**2 + 0.5 * yy**2
    f2_data_full = f2_grid.reshape(-1, 1)
    
    # 2. Format calibration data for the NN
    x_cal = x_data_full
    y_cal = f2_data_full

    return x_data_full, f2_data_full, xx, yy, f2_grid, x_cal, y_cal



# ----- Data generation function for the gradient of f2 ----- #
"""Tasks: 2.1, 2.3"""


def grad_f2():
    """Generates data for the gradient of f2(x)"""
    # 1. Generate the full (20x20) grid for plotting
    x_data_full, xx, yy, _, _ = generate_2d_grid()
    grad_x_grid = 2 * xx
    grad_y_grid = yy
    grad_f2_data_full = jnp.stack([
        grad_x_grid.flatten(), 
        grad_y_grid.flatten()
    ], axis=1)

    # 2. Format calibration data for the NN
    x_cal = x_data_full
    y_cal = grad_f2_data_full
    
    return x_data_full, grad_f2_data_full, xx, yy, grad_x_grid, grad_y_grid, x_cal, y_cal
    


# ----- Helper plotting function for 3D surface ----- #
"""Tasks: 2.1"""


def plot_3d_surface(xx, yy, z_grid, title):
    """
    Plots a 2D function z = f(x, y) as a 3D surface
    with a colorbar.
    
    Args:
        xx: (N, N) meshgrid for x-coordinates
        yy: (N, N) meshgrid for y-coordinates
        z_grid: (N, N) grid of z values (the function output)
        title: Title for the plot
    """
    fig = plt.figure(figsize=(10, 8)) # Made figure a bit larger
    ax = fig.add_subplot(projection='3d')
    
    # 1. Capture the plot object
    plot_obj = ax.plot_surface(xx, yy, z_grid, cmap='viridis', rcount=20, ccount=20)
    
    # 2. Add a colorbar linked to the plot object
    fig.colorbar(plot_obj, shrink=0.5, aspect=5, label='f(x, y) value')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title(title)
    plt.show()



# ----- Helper plotting function for plotting 2D gradients as a vector field ----- #
"""Tasks: 2.1"""


def plot_2d_vector_field(xx, yy, u_grid, v_grid, title):
    """
    Plots a 2D vector field F = (u, v) as a quiver plot.
    
    Args:
        xx: (N, N) meshgrid for x-coordinates
        yy: (N, N) meshgrid for y-coordinates
        u_grid: (N, N) grid of u-component of the vector
        v_grid: (N, N) grid of v-component of the vector
        title: Title for the plot
    """
    plt.figure(figsize=(8, 7))
    plt.quiver(xx, yy, u_grid, v_grid)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.show()



# ----- Helper function used in plot_3d_overlay function ----- #
"""Tasks: 2.1"""


def get_ensemble_predictions_2d(trained_ensemble, x_data):
    """
    Calculates the mean and std dev of predictions from a 2D-input ensemble.
    
    Args:
        trained_ensemble: A list of trained klax models.
        x_data: The (N, 2) input data grid.
    
    Returns:
        mean_preds: (N, 1) mean of the predictions.
        std_preds: (N, 1) std dev of the predictions.
    """
    # 1. Finalize all models and create vmapped versions
    #    Note: jax.vmap is for batching, not for the ensemble
    finalized_models = [jax.vmap(klax.finalize(m)) for m in trained_ensemble]
    
    # 2. Get predictions from every model in the ensemble
    #    We use a list comprehension, not vmap, for the ensemble
    predictions = [m(x_data) for m in finalized_models]
    
    # 3. Stack predictions into (ensemble_size, num_data_points, 1)
    predictions_stack = jnp.stack(predictions)
    
    # 4. Calculate mean and std dev across the ensemble axis (axis=0)
    mean_preds = jnp.mean(predictions_stack, axis=0)
    std_preds = jnp.std(predictions_stack, axis=0)
    
    return mean_preds, std_preds



# ----- 3D plotting function showing true data, NN predicted function and calibration data in one plot ----- #
"""Tasks: 2.2"""


def plot_3d_overlay(
    xx, yy, z_true_grid, 
    trained_ensemble, x_data_full, 
    x_cal, y_cal, 
    title
):
    """
    Plots the true function, model wireframe, and calibration points
    on a single 3D plot.
    """
    # Get mean predictions from the ensemble on the FULL grid
    mean_preds, _ = get_ensemble_predictions_2d(trained_ensemble, x_data_full)
    z_pred_grid = mean_preds.reshape(xx.shape)
    
    # Get z-values for the calibration points
    z_cal = y_cal.flatten()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection='3d')

    # 1. Plot the True Function as a semi-transparent surface
    # --- MODIFICATION: Capture the surface object 'surf' ---
    surf = ax.plot_surface(
        xx, yy, z_true_grid, 
        cmap='viridis', 
        alpha=0.6
        # The label is now handled by the colorbar
    )
    
    # --- MODIFICATION: Add a colorbar for the surface ---
    fig.colorbar(surf, shrink=0.5, aspect=10, label='True Function Value')

    # 2. Plot the Model Prediction as a wireframe
    ax.plot_wireframe(
        xx, yy, z_pred_grid, 
        color='red', 
        rstride=1, cstride=1, 
        linewidth=0.8, 
        label="Model Prediction (Mean)" # This label will be used
    )

    # 3. Plot the Calibration Points
    ax.scatter(
        x_cal[:, 0], x_cal[:, 1], z_cal, 
        color='black', 
        depthshade=False, 
        s=40, 
        label="Calibration Data" # This label will be used
    )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title(title)
    
    # --- MODIFICATION: Call the standard legend ---
    # This will automatically pick up the 'label' from the
    # wireframe and scatter plots.
    ax.legend()
    
    plt.show()


# ----- Simple helper function to package data from f2 and it's gradient for Sobolev training ----- #
"""Tasks: 2.3"""


def f2_and_grad():
    """
    Generates data for f2 and its gradient, packaged for
    Sobolev training.
    
    This version calls the simple f2() and grad_f2() functions
    that use the full grid by default.
    """
    # 1. Get the f2 data
    (x_data_full, y_data_full, 
     xx, yy, z_true_grid, 
     x_cal, y_cal) = f2()
    
    # 2. Get the grad_f2 data
    (_, _, _, _, grad_x_true_grid, grad_y_cal_grid, _, y_grad_cal) = grad_f2()
    
    # 3. Package the y_data as a tuple
    y_cal_combined = (y_cal, y_grad_cal)
    
    # Return everything needed for training and plotting
    # (Note: grad_f2 returns grad_y_cal_grid, but we need grad_x_true_grid
    #  from the full grid for plotting, so we call grad_f2 once more
    #  and just grab the grid)
    (_, _, _, _, grad_x_true_grid, grad_y_true_grid, _, _) = grad_f2()
    
    return (x_data_full, y_data_full, 
            xx, yy, z_true_grid, 
            x_cal, y_cal_combined,
            grad_x_true_grid, grad_y_true_grid) # <-- Return grid for plotting



# ----- Helper function used in plot_3d_sobolev_overlay, plot_2d_sobolev_grad_comparison, plot_1d_stress_strain, calculate_and_print_errors functions ----- #
"""Tasks: 2.3, 2.4"""


def get_ensemble_predictions_sobolev(trained_ensemble, x_data):
    """
    Calculates the mean and std dev of predictions from a Sobolev ensemble.
    
    Args:
        trained_ensemble: A list of trained SobolevModel instances.
        x_data: The (N, 1) or (N, 2) input data grid.
    
    Returns:
        A tuple of (mean_values, std_values, mean_grads, std_grads).
    """
    value_preds_list = []
    grad_preds_list = []
    
    # We vmap the finalized model once for each ensemble member
    vmapped_models = [jax.vmap(klax.finalize(m)) for m in trained_ensemble]
    
    for vmapped_model in vmapped_models:
        # model call returns a tuple: (values, grads)
        values, grads = vmapped_model(x_data)
        value_preds_list.append(values)
        grad_preds_list.append(grads)
        
    # Stack along the new ensemble axis (axis=0)
    values_stack = jnp.stack(value_preds_list, axis=0) 
    grads_stack = jnp.stack(grad_preds_list, axis=0)   
    
    # Get mean
    mean_values = jnp.mean(values_stack, axis=0)
    mean_grads = jnp.mean(grads_stack, axis=0)
    
    # Get std dev
    std_values = jnp.std(values_stack, axis=0)
    std_grads = jnp.std(grads_stack, axis=0)
    
    return mean_values, std_values, mean_grads, std_grads



# ----- 3D plotting function showing true data, NN predicted function and calibration data in one plot for SobolevModels ----- #
"""Tasks: 2.3"""


def plot_3d_sobolev_overlay(
    xx, yy, z_true_grid, 
    trained_ensemble, x_data_full, 
    x_cal, y_cal_true, 
    title
):
    """
    Plots the f2 value prediction from a Sobolev ensemble.
    """
    # 1. Get mean predictions from the ensemble
    mean_values, _, _, _ = get_ensemble_predictions_sobolev(trained_ensemble, x_data_full)
    
    z_pred_grid = mean_values.reshape(xx.shape)
    
    # 2. Get z-values for the calibration points
    z_cal = y_cal_true.flatten()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection='3d')

    # ... (rest of the plotting function is unchanged) ...
    surf = ax.plot_surface(
        xx, yy, z_true_grid, 
        cmap='viridis', 
        alpha=0.6
    )
    fig.colorbar(surf, shrink=0.5, aspect=10, label='True Function Value')
    ax.plot_wireframe(
        xx, yy, z_pred_grid, 
        color='red', 
        rstride=1, cstride=1, 
        linewidth=0.8, 
        label="Model Prediction (Mean)"
    )
    ax.scatter(
        x_cal[:, 0], x_cal[:, 1], z_cal, 
        color='black', 
        depthshade=False, 
        s=40, 
        label="Calibration Data"
    )
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f(x, y)')
    ax.set_title(title)
    ax.legend()
    plt.show()



# ----- Helper plotting function for plotting 2D gradient predictions of NN as a vector field with relative error distribution heatmap ----- #
"""Tasks: 2.3"""


def plot_2d_sobolev_grad_comparison(
    xx, yy, grad_x_true_grid, grad_y_true_grid,
    trained_ensemble, x_data_full,
    title
):
    """
    Plots the predicted gradient field with a heatmap of the
    RELATIVE error magnitude.
    """
    # 1. Get mean gradient predictions
    _, _, mean_grads, _ = get_ensemble_predictions_sobolev(trained_ensemble, x_data_full)
    
    # 2. Reshape predictions
    u_pred_grid = mean_grads[:, 0].reshape(xx.shape)
    v_pred_grid = mean_grads[:, 1].reshape(xx.shape)

    # 3. Calculate Relative Error Magnitude
    true_grad = jnp.stack([grad_x_true_grid, grad_y_true_grid], axis=-1)
    pred_grad = jnp.stack([u_pred_grid, v_pred_grid], axis=-1)
    
    true_mag = jnp.linalg.norm(true_grad, axis=-1)
    error_mag = jnp.linalg.norm(true_grad - pred_grad, axis=-1)
    
    # Calculate relative error, add a small epsilon to avoid divide-by-zero
    relative_error_grid = error_mag / (true_mag + 1e-8)

    # 4. Create the *single* plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot the error magnitude as a color heatmap
    # We set vmin/vmax to 0 and 1 for a 0-100% relative error scale
    contour = ax.contourf(
        xx, yy, relative_error_grid, 
        cmap='Reds', 
        levels=jnp.linspace(0, 1.0, 21), # 0% to 100% in 5% steps
        alpha=0.8
    )
    fig.colorbar(contour, ax=ax, label='Relative Gradient Error Magnitude (%)')
    
    # Plot the predicted vector field on top
    ax.quiver(xx, yy, u_pred_grid, v_pred_grid, color='black', alpha=0.8)
    
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f"Model Prediction (Mean Gradient): {title}")
    ax.grid(True)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    plt.tight_layout()
    plt.show()



# ----- Function to calculate some summary statistics for quantitative analysis ----- #
"""Tasks: 2.3"""


def calculate_and_print_errors(
    trained_ensemble,
    x_data_full,
    y_data_full,
    grad_x_true_grid,
    grad_y_true_grid,
    strategy_name
):
    """
    Calculates and prints expanded error statistics (MSE, MAE, Median, Max)
    for both function value and gradient predictions.
    """
    
    # --- 1. Get Mean Predictions ---
    mean_values, _, mean_grads, _ = get_ensemble_predictions_sobolev(
        trained_ensemble, x_data_full
    )
    
    # --- 2. Prepare True Data ---
    y_true_squeezed = y_data_full.squeeze()
    grad_true = jnp.stack([
        grad_x_true_grid.flatten(), 
        grad_y_true_grid.flatten()
    ], axis=1)

    # --- 3. Calculate Errors ---
    
    # --- Value Errors ---
    value_abs_errors = jnp.abs(y_true_squeezed - mean_values)
    value_mse = jnp.mean(value_abs_errors**2)
    value_mae = jnp.mean(value_abs_errors)
    value_median = jnp.median(value_abs_errors) # <-- NEW
    value_max = jnp.max(value_abs_errors)       # <-- NEW
    
    # --- Gradient Errors ---
    grad_abs_errors = jnp.linalg.norm(grad_true - mean_grads, axis=-1)
    grad_mse = jnp.mean(grad_abs_errors**2)
    grad_mae = jnp.mean(grad_abs_errors)
    grad_median = jnp.median(grad_abs_errors)   # <-- NEW
    grad_max = jnp.max(grad_abs_errors)         # <-- NEW

    # --- 4. Print Results ---
    print(f"--- Quantitative Errors for Strategy: {strategy_name} ---")
    print(f"  f(x) Value Prediction (vs. true f2):")
    print(f"    - MSE:    {value_mse:10.6f}")
    print(f"    - MAE:    {value_mae:10.6f}")
    print(f"    - Median: {value_median:10.6f}")
    print(f"    - Max:    {value_max:10.6f}")
    print(f"  ∇f(x) Gradient Prediction (vs. true ∇f2):")
    print(f"    - MSE:    {grad_mse:10.6f}")
    print(f"    - MAE:    {grad_mae:10.6f}")
    print(f"    - Median: {grad_median:10.6f}")
    print(f"    - Max:    {grad_max:10.6f}")
    print("--------------------------------------------------\n")



# ----- Data generation function for 1D stress strain problem in task 2.4 ----- #
"""Tasks: 2.4"""


def stress_strain_1d():
    """
    Loads the 1D stress-strain data for the hyperelasticity problem provided to us via email.
    Splits the data into a calibration set (eps < 5) and a
    test/extrapolation set (eps >= 5).
    """
    eps_list = [1.0704, 1.141, 1.2118, 1.2826, 1.3532, 1.4244, 1.495, 1.5662, 1.637, 1.708, 1.7786, 1.8494, 1.9202, 1.991, 2.062, 2.1326, 2.2032, 2.274, 2.3446, 2.4156, 2.4864, 2.5574, 2.628, 2.6988,
        2.7694, 2.8402, 2.9112, 2.9818, 3.0526, 3.1234, 3.1942, 3.265, 3.336, 3.4066, 3.4774, 3.5482, 3.619, 3.69, 3.7606, 3.8312, 3.902, 3.9726, 4.0434, 4.1142, 4.1854, 4.2558, 4.3266, 4.3976, 4.4682, 4.5392, 4.61,
        4.681, 4.7516, 4.8224, 4.893, 4.9638, 5.0346, 5.1052, 5.1758, 5.2468, 5.3172, 5.388, 5.4588, 5.5296, 5.6006, 5.6712, 5.7416, 5.8122, 5.8826, 5.9534, 6.0242, 6.095, 6.1656, 6.2362, 6.307, 6.3776, 6.4486, 6.5194,
        6.5902, 6.6612, 6.7316, 6.8026, 6.8736, 6.9444]
    sigma_list = [0.0088, 0.0171, 0.02528, 0.02985, 0.03671, 0.04224, 0.04597, 0.05169, 0.05701, 0.06111, 0.06718, 0.07141, 0.07732, 0.08283, 0.08914, 0.09536, 0.10144, 0.10905, 0.11511, 0.12411, 0.13149, 0.14099, 0.14887, 0.15849,
            0.16889, 0.17946, 0.19119, 0.20207, 0.21451, 0.22583, 0.23752, 0.254, 0.26501, 0.27982, 0.29696, 0.30841, 0.32479, 0.34257, 0.35524, 0.37358, 0.39065, 0.40596, 0.42616, 0.44378, 0.4608, 0.47952, 0.49956, 0.51609,
            0.53559, 0.55574, 0.57587, 0.5959, 0.61616, 0.63696, 0.65799, 0.67913, 0.70165, 0.72271, 0.74524, 0.76824, 0.78911, 0.813, 0.83704, 0.86004, 0.88422, 0.90847, 0.93198, 0.95591, 0.9805, 1.00463, 1.02894, 1.05455,
            1.07938, 1.1037, 1.12929, 1.1546, 1.17734, 1.20359, 1.23015, 1.25425, 1.2796, 1.30502, 1.32963, 1.35553]


    # Convert to JAX arrays
    eps = jnp.array(eps_list)
    sigma = jnp.array(sigma_list)

    # Find the split index based on the condition
    split_index = jnp.sum(eps < 5)

    # --- Create Calibration Set (eps < 5) ---
    eps_cal = eps[:split_index].reshape(-1, 1)
    sigma_cal = sigma[:split_index].reshape(-1, 1)

    # --- Create Test/Extrapolation Set (eps >= 5) ---
    eps_test = eps[split_index:].reshape(-1, 1)
    sigma_test = sigma[split_index:].reshape(-1, 1)

    # Also return the full data for plotting
    eps_full = eps.reshape(-1, 1)
    sigma_full = sigma.reshape(-1, 1)
    
    return eps_full, sigma_full, eps_cal, sigma_cal, eps_test, sigma_test
            


# ----- Simple 1D plotting function for sobolev stress strain model in task 2.4 ----- #
"""Tasks: 2.4"""


def plot_1d_stress_strain(
    eps_full, sigma_full,
    eps_cal, sigma_cal,
    trained_ensemble, title
):
    """
    Plots the true stress-strain curve and the predicted
    stress-strain curve from a Sobolev-trained ensemble.
    """
    
    # 1. Get mean predictions and std dev from the ensemble
    _, _, mean_grads, std_grads = get_ensemble_predictions_sobolev(
        trained_ensemble, eps_full
    )
    
    # --- Create the plot (this part is now correct) ---
    plt.figure(figsize=(10, 6))
    
    # Plot the full true data
    plt.plot(eps_full, sigma_full, c="black", linestyle="--", label="True Stress (Data)")
    
    # Plot the calibration points
    plt.scatter(eps_cal, sigma_cal, c="green", label="Calibration Data", alpha=0.7, zorder=3)
    
    # Plot the mean prediction
    plt.plot(eps_full, mean_grads, label="Model Prediction (Mean Stress)", color="red", zorder=5)

    # Plot the uncertainty region
    plt.fill_between(
        eps_full.flatten(),
        (mean_grads - 2 * std_grads).flatten(),
        (mean_grads + 2 * std_grads).flatten(),
        color="red",
        alpha=0.2,
        label="Ensemble Uncertainty (±2σ)",
        zorder=2
    )
    
    plt.title(f"Stress-Strain Prediction: {title}")
    plt.xlabel("Strain (ε)")
    plt.ylabel("Stress (σ)")
    plt.legend()
    plt.grid(True)
    plt.show()