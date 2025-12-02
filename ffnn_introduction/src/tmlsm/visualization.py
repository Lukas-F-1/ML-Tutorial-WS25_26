import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import jax.numpy as jnp
import jax

def visualize_deformation_3d(F, step_index=0):
    """
    Visualisiert die Deformation eines Einheitswürfels für einen spezifischen Zeitschritt.
    
    Parameters
    ----------
    F : ndarray, shape (N, 3, 3)
        Deformationsgradienten über die Zeit.
    step_index : int
        Der Index (Zeitpunkt), der visualisiert werden soll.
    """
    # 1. Extrahiere F für den gewählten Schritt
    if step_index >= len(F) or step_index < 0:
        raise ValueError(f"step_index {step_index} ist außerhalb des Bereichs (0 bis {len(F)-1})")
    
    Fn = F[step_index]
    
    # 2. Definiere die Eckpunkte des unverformten Einheitswürfels (Referenzkonfiguration)
    # Eckpunkte: (x, y, z)
    points_ref = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], # Unten (0-3)
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]  # Oben (4-7)
    ])
    
    # 3. Berechne die verformten Eckpunkte: x = F * X
    points_def = np.dot(points_ref, Fn.T)

    # Hilfsfunktion, um Flächen aus Eckpunkten zu definieren
    def get_faces(points):
        # Die Indizes beziehen sich auf die Reihenfolge in 'points_ref'
        faces = [
            [points[0], points[1], points[5], points[4]], # Front
            [points[7], points[6], points[2], points[3]], # Back
            [points[0], points[3], points[7], points[4]], # Left
            [points[1], points[2], points[6], points[5]], # Right
            [points[0], points[1], points[2], points[3]], # Bottom
            [points[4], points[5], points[6], points[7]]  # Top
        ]
        return faces

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- Plot Referenz (Unverformt) ---
    faces_ref = get_faces(points_ref)
    # Wireframe (Kanten)
    for face in faces_ref:
        x, y, z = zip(*face)
        # Schließe den Loop für die Linie
        x = list(x) + [x[0]]
        y = list(y) + [y[0]]
        z = list(z) + [z[0]]
        # Jetzt mit roter Farbe für die Kanten des Referenzwürfels!
        ax.plot(x, y, z, color='red', linestyle='-', linewidth=2, alpha=0.8)
    
    # --- Plot Aktuell (Verformt) ---
    faces_def = get_faces(points_def)
    
    # Transparente Flächen
    mesh = Poly3DCollection(faces_def, alpha=0.3, edgecolor='k') # Schwarze Kanten für den verformten Würfel
    mesh.set_facecolor('cyan')
    ax.add_collection3d(mesh)

    # --- Achsen-Einstellungen ---
    
    # Titel mit Info über die Diagonalelemente
    diag_vals = np.diag(Fn)
    title_str = (f"Deformation bei Schritt {step_index}\n"
                 f"F_diag = [{diag_vals[0]:.2f}, {diag_vals[1]:.2f}, {diag_vals[2]:.2f}]")
    ax.set_title(title_str)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # WICHTIG: Skalierung fixieren, damit Würfel nicht verzerrt wirken
    all_points = np.vstack((points_ref, points_def))
    max_range = np.array([all_points[:,0].max()-all_points[:,0].min(), 
                          all_points[:,1].max()-all_points[:,1].min(), 
                          all_points[:,2].max()-all_points[:,2].min()]).max() / 2.0

    mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
    mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
    mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

def plot_F_diagonals(F, time=None, components=["F11", "F22", "F33"], title="Diagonal entries of F"):
    """
    Plot selected diagonal components of F.

    Parameters
    ----------
    F : ndarray, shape (N, 3, 3)
        Sequence of deformation gradients.
    time : array-like, optional
        Time steps. Defaults to step index.
    components : list of str, optional
        List of components to plot. Options: "F11", "F22", "F33".
        Default is ["F11", "F22", "F33"] (plots all).
    title : str, optional
        Title of the plot.
    """
    F = np.asarray(F)
    if F.ndim != 3 or F.shape[1:] != (3, 3):
        raise ValueError("F must have shape (N, 3, 3)")

    n_steps = F.shape[0]

    if time is None:
        time = np.arange(n_steps)
    else:
        time = np.asarray(time)

# Dictionary mapping labels to the actual data slices
    data_map = {
        # Zeile 1
        "F11": F[:, 0, 0],
        "F12": F[:, 0, 1],
        "F13": F[:, 0, 2],
        
        # Zeile 2
        "F21": F[:, 1, 0],
        "F22": F[:, 1, 1],
        "F23": F[:, 1, 2],
        
        # Zeile 3
        "F31": F[:, 2, 0],
        "F32": F[:, 2, 1],
        "F33": F[:, 2, 2]
    }

    fig, ax = plt.subplots()

    # Loop through the requested components and plot them
    for comp in components:
        if comp in data_map:
            ax.plot(time, data_map[comp], label=comp)
        else:
            print(f"Warnung: '{comp}' ist keine gültige Komponente (erlaubt: F11, F22, F33).")

    ax.set_xlabel("Step / time")
    ax.set_ylabel("Diagonal entries of F")
    ax.set_title(title)
    ax.grid(True)
    
    # Only show legend if we actually plotted something
    if components:
        ax.legend()

    fig.tight_layout()
    plt.show()

    return fig, ax

def plot_model_and_history(model, X_cal, Y_cal, history, *,
                           title_model="Model Prediction",
                           title_history="Training History"):
    """
    Plots:
    1. Model predictions vs ground truth (for calibration data)
    2. Training loss over iterations

    Parameters
    ----------
    model : trained model (from tm.train_model)
    X_cal : input data used for training  (shape: (N, input_dim))
    Y_cal : ground truth output          (shape: (N, output_dim))
    history : klax training history object
    """

    # -----------------------
    # Compute model predictions
    # -----------------------
    Y_pred = jax.vmap(model)(X_cal)

    # -----------------------
    # FIGURE 1: Model predictions
    # -----------------------
    plt.figure(figsize=(10, 4))
    plt.plot(Y_cal[:, 0],    label="Ground Truth", linewidth=2)
    plt.plot(Y_pred[:, 0],  label="Prediction", linestyle="--")
    plt.title(title_model)
    plt.xlabel("Sample Index")
    plt.ylabel("Output Component 0")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # NOTE:
    # Plotting component 0 is just an example.
    # You can loop over components if needed.

    # -----------------------
    # FIGURE 2: Training loss
    # -----------------------
    plt.figure(figsize=(10, 4))
    plt.plot(history.loss, linewidth=2)
    plt.yscale("log")
    plt.title(title_history)
    plt.xlabel("Training Step")
    plt.ylabel("Loss (log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax


def evaluate_MS_predictions(Y_true, Y_pred, title_prefix="MS Test Evaluation"):
    """
    Visualizes:
    1. Pred vs True for all 9 components
    2. Component-wise error plots
    3. Frobenius error measure
    4. True vs Pred scatter plot grid
    """

    N = Y_true.shape[0]

    # --------------------------------------------
    # 1. Prediction vs True (9 components)
    # --------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]
        ax.plot(Y_true[:, i], label="True", linewidth=2)
        ax.plot(Y_pred[:, i], label="Pred", linestyle='--')
        ax.set_title(f"P component {i}")
        ax.grid(True)
        if i == 0:
            ax.legend()

    plt.suptitle(f"{title_prefix}: Prediction vs True")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------
    # 2. Component-wise error plots
    # --------------------------------------------
    errors = Y_pred - Y_true

    plt.figure(figsize=(12, 4))
    for i in range(9):
        plt.plot(errors[:, i], label=f"Comp {i}")
    plt.title(f"{title_prefix}: Component Errors")
    plt.xlabel("Sample Index")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------
    # 3. Frobenius norm error
    # --------------------------------------------
    E_frob = jnp.sqrt(jnp.sum(errors**2, axis=1))

    plt.figure(figsize=(10, 4))
    plt.plot(E_frob, linewidth=2)
    plt.title(f"{title_prefix}: Frobenius Norm Error")
    plt.xlabel("Sample Index")
    plt.ylabel("||Error||_F")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------
    # 4. Scatter: Pred vs True (all components)
    # --------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]
        ax.scatter(Y_true[:, i], Y_pred[:, i], s=4)
        ax.plot([Y_true[:, i].min(), Y_true[:, i].max()],
                [Y_true[:, i].min(), Y_true[:, i].max()],
                'r--')
        ax.set_title(f"Scatter P[{i}]")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.grid(True)

    plt.suptitle(f"{title_prefix}: Scatter True vs Pred")
    plt.tight_layout()
    plt.show()

    # return error metrics if needed
    return {
        "mae": jnp.mean(jnp.abs(errors)),
        "rmse": jnp.sqrt(jnp.mean(errors**2)),
        "max_error": jnp.max(jnp.abs(errors)),
        "frob_mean": jnp.mean(E_frob),
    }
