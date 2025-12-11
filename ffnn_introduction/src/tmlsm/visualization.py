import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import jax.numpy as jnp
import jax
from . import data_t2 as td2

def plot_load_path_space(datasets_dict, title="Load Path Map: Interpolation Check"):
    """
    Visualizes the load paths in three projections:
    1. F11 vs F22
    2. F11 vs F33
    3. F22 vs F33
    
    Arranged side-by-side in one row.
    """
    # Create 3 subplots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5)) 
    
    # Define the pairs we want to plot (Index 0=F11, 1=F22, 2=F33)
    # Format: (x_idx, y_idx, x_label, y_label)
    plot_configs = [
        (0, 1, r"$F_{11}$ (Fiber)", r"$F_{22}$ (Transverse)"),
        (0, 2, r"$F_{11}$ (Fiber)", r"$F_{33}$ (Normal)"),
        (1, 2, r"$F_{22}$ (Transverse)", r"$F_{33}$ (Normal)")
    ]

    # Iterate over datasets
    for label, data in datasets_dict.items():
        # --- ROBUST UNPACKING ---
        if isinstance(data, (tuple, list)):
            F = data[0]
        else:
            F = data
            
        # Style logic
        is_test = "Test" in label
        style = '--' if is_test else '-'
        alpha = 1.0 if is_test else 0.6
        width = 2.5 if is_test else 1.5

        # Loop through the 3 subplots
        for ax, (x_idx, y_idx, x_lbl, y_lbl) in zip(axes, plot_configs):
            x_data = F[:, x_idx, x_idx] # Diagonal element
            y_data = F[:, y_idx, y_idx] # Diagonal element
            
            ax.plot(x_data, y_data, linestyle=style, linewidth=width, alpha=alpha, label=label)

    # Styling for each subplot
    for i, ax in enumerate(axes):
        # Reference point (1,1)
        ax.scatter([1], [1], color='black', marker='x', s=80, zorder=10)
        
        # Labels from config
        ax.set_xlabel(plot_configs[i][2], fontsize=12)
        ax.set_ylabel(plot_configs[i][3], fontsize=12)
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.axis('equal') # Keep geometric proportions
        
        # Only show legend on the first plot to avoid clutter, 
        # or put it outside. Let's put it on the last one or first.
        if i == 0:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_deformation_state_space(datasets_dict, components=None, title="State Space: Shear vs. Stretch Intensity"):
    """
    Visualizes the distribution of F-components using Boxplots in a 3x3 grid
    matching the tensor structure of F.
    
    Args:
        datasets_dict: Dictionary with data.
        components: List of strings determining which tensor components to plot.
                    If None, all 9 components are plotted.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Default: alle 9 Komponenten
    if components is None:
        components = ["F11", "F12", "F13", "F21", "F22", "F23", "F31", "F32", "F33"]
    
    # Mapping string "F12" -> indices (row, col) im Grid und im Tensor
    comp_map = {
        "F11": (0, 0), "F12": (0, 1), "F13": (0, 2),
        "F21": (1, 0), "F22": (1, 1), "F23": (1, 2),
        "F31": (2, 0), "F32": (2, 1), "F33": (2, 2)
    }
    
    # Kurzbezeichnungen und Farben für die Loadpaths
    label_abbrev = {
        "Train: Uniaxial": "Uni",
        "Train: Pure Shear": "PS",
        "Train: Biaxial": "Bi",
        "Test: Biaxial": "Bi*",
        "Test: Mixed": "Mix*"
    }
    
    # Farbpalette (konsistent für alle Plots)
    palette = sns.color_palette("tab10", n_colors=len(datasets_dict))
    color_map = {label: palette[i] for i, label in enumerate(datasets_dict.keys())}
    
    # 1. Daten vorbereiten und globale Y-Grenzen bestimmen
    records = {comp: [] for comp in comp_map.keys()}
    global_min, global_max = float('inf'), float('-inf')
    
    for label, data in datasets_dict.items():
        F = data[0] if isinstance(data, (tuple, list)) else data
        category = "Test" if "Test" in label else "Train"
        abbrev = label_abbrev.get(label, label[:3])
        
        # Globale Min/Max tracken
        global_min = min(global_min, F.min())
        global_max = max(global_max, F.max())
        
        for comp_name, (i, j) in comp_map.items():
            values = F[:, i, j]
            for val in values:
                records[comp_name].append({
                    "Dataset": label,
                    "Abbrev": abbrev,
                    "Type": category,
                    "Value": float(val)
                })
    
    # Etwas Padding für die Y-Achse
    y_padding = (global_max - global_min) * 0.05
    y_limits = (global_min - y_padding, global_max + y_padding)
    
    # 2. 3x3 Grid erstellen
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    
    for comp_name, (row, col) in comp_map.items():
        ax = axes[row, col]
        
        # DataFrame für diese Komponente
        df_comp = pd.DataFrame(records[comp_name])
        
        if comp_name in components and not df_comp.empty:
            # Boxplot mit Kurzbezeichnungen
            sns.boxplot(data=df_comp, x="Abbrev", y="Value", hue="Dataset",
                       palette=color_map, ax=ax, width=0.7, dodge=False,
                       legend=False)
            
            # Test-Punkte als Overlay
            test_df = df_comp[df_comp["Type"] == "Test"]
            if not test_df.empty:
                sns.stripplot(data=test_df, x="Abbrev", y="Value",
                            color="black", alpha=0.3, ax=ax, size=2)
            
            ax.set_title(f"$F_{{{row+1}{col+1}}}$", fontsize=14, fontweight='bold')
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_ylim(y_limits)  # Einheitliche Y-Skala
            
            # X-Achsen Labels nur in der untersten Reihe
            if row < 2:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis='x', labelsize=10)
                
        else:
            ax.set_visible(False)
        
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
    
    # 3. Gemeinsame Legende erstellen
    legend_patches = [mpatches.Patch(color=color_map[label], 
                                      label=f"{label_abbrev.get(label, label)} = {label}")
                     for label in datasets_dict.keys()]
    
    fig.legend(handles=legend_patches, 
              loc='lower center', 
              ncol=3,
              fontsize=11,
              framealpha=0.9,
              bbox_to_anchor=(0.5, -0.02))
    
    # Gemeinsame Y-Achsenbeschriftung
    fig.text(0.02, 0.5, 'Value', va='center', rotation='vertical', fontsize=12)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0.04, 0.08, 1, 0.96])
    plt.show()

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

def plot_generalization_heatmap(results_df, title="Model Generalization: Test Error Heatmap"):
    """
    Plots a vertically split heatmap comparing two models across different
    training set sizes and multiple runs.
    
    Args:
        results_df: DataFrame with columns:
            - 'Model': str ('Naive FFNN' or 'PANN')
            - 'N_Train_Paths': int (number of training loadpaths)
            - 'Run': int (run index 0-4)
            - 'Test_Error': float (RMSE or similar metric)
        title: str, plot title
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Pivot für beide Modelle
    models = ['Naive FFNN', 'PANN']
    n_paths_list = sorted(results_df['N_Train_Paths'].unique())
    n_runs = results_df['Run'].nunique()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Gemeinsame Farbskala bestimmen (log-scale friendly)
    vmin = results_df['Test_Error'].min()
    vmax = results_df['Test_Error'].max()
    
    for ax, model_name in zip(axes, models):
        # Daten für dieses Modell filtern und pivotieren
        model_df = results_df[results_df['Model'] == model_name]
        pivot = model_df.pivot(index='Run', columns='N_Train_Paths', values='Test_Error')
        
        # Sicherstellen dass Spalten sortiert sind
        pivot = pivot.reindex(columns=n_paths_list)
        
        # Heatmap
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r',
                       vmin=vmin, vmax=vmax)
        
        # Werte in Zellen schreiben
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    # Textfarbe basierend auf Hintergrund
                    text_color = 'white' if val > (vmax + vmin) / 2 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                           fontsize=9, color=text_color, fontweight='bold')
        
        # Y-Achse: Runs
        ax.set_yticks(range(n_runs))
        ax.set_yticklabels([f'Run {r+1}' for r in range(n_runs)])
        ax.set_ylabel(model_name, fontsize=12, fontweight='bold')
        
        # Trennlinie-Effekt durch Rahmen
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    
    # X-Achse nur unten
    axes[1].set_xticks(range(len(n_paths_list)))
    axes[1].set_xticklabels([str(n) for n in n_paths_list])
    axes[1].set_xlabel('Number of Training Loadpaths', fontsize=12)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Test Error (RMSE)', fontsize=11)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig

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

def plot_loadpath(F_path, P_path, W_path, title_prefix="Loadpath"):
    """
    Produces 3 figures:
      1. Principal stretches
      2. Strain energy
      3. 3×3 grid of stress tensor components
      
    Parameters
    ----------
    F_path : array, shape (T,3,3)
        Deformation gradients along the loadpath
    P_path : array, shape (T,3,3)
        First Piola stress tensors
    W_path : array, shape (T,)
        Strain energies
    title_prefix : str
        Label printed on top of each plot
    """
    
    # -------------------------------------------------------
    # 1. Principal stretches: eigenvalues of C = FᵀF
    # -------------------------------------------------------
    C_path = jnp.einsum("tij,tkj->tik", F_path.transpose(0,2,1), F_path)
    evals = jnp.linalg.eigvalsh(C_path)      # symmetric eigenvalues
    lambdas = jnp.sqrt(evals)                # principal stretches

    plt.figure(figsize=(7,5))
    plt.plot(lambdas[:,0], label="λ₁")
    plt.plot(lambdas[:,1], label="λ₂")
    plt.plot(lambdas[:,2], label="λ₃")
    plt.title(f"{title_prefix} – Principal Stretches")
    plt.xlabel("Time step")
    plt.ylabel("Stretch")
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------------------------------------------------------
    # 2. Strain energy
    # -------------------------------------------------------
    plt.figure(figsize=(7,5))
    plt.plot(W_path, linewidth=2)
    plt.title(f"Energy – {title_prefix}")
    plt.xlabel("Time step")
    plt.ylabel("Strain energy W")
    plt.grid(True)
    plt.show()

    # -------------------------------------------------------
    # 3. Stress components (3×3 grid)
    # -------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()

    P_flat = P_path.reshape(len(P_path), 9)

    for i in range(9):
        ax = axes[i]
        ax.plot(P_flat[:, i], linewidth=2)
        ax.set_title(f"P component {i}")
        ax.grid(True)

    plt.suptitle(f"Stress Components – {title_prefix}", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_dataset_distributions(W_all, P_all, bins=100):
    """
    Visualizes global distributions of strain energy W and stress tensor P.
    
    Parameters:
    -----------
    W_all : array (N,)
        All strain energy values.
    P_all : array (N, 3, 3)
        All first Piola–Kirchhoff stress tensors.
    bins : int
        Number of histogram bins.
    """

    # Flatten stress to shape (N, 9)
    P_flat = P_all.reshape(-1, 9)

    # ---------------------------
    # 1. Histogram for W
    # ---------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(W_all, bins=bins, color="steelblue", alpha=0.75)
    plt.title("Distribution of Strain Energy W")
    plt.xlabel("W")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # ---------------------------
    # 2. Histograms for P components
    # ---------------------------
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]
        ax.hist(P_flat[:, i], bins=bins, alpha=0.75, color="salmon")
        ax.set_title(f"P component {i}")
        ax.grid(True)

    plt.suptitle("Distribution of Stress Components P_ij")
    plt.tight_layout()
    plt.show()

def evaluate_model_performance(model, inputs, P_true, W_true=None, history=None, title="Model Evaluation"):

    """
    A universal function for evaluating models (Naive & PANN).
    
    Features:
    - Automatically generates predictions (handles both Naive and PANN structures).
    - Detects whether Strain Energy (W) is predicted.
    - Plots Training Loss History (optional).
    - Plots Energy Comparison (optional).
    - Plots Stress Comparison (always).
    
    Parameters
    ----------
    model : Callable
        The trained model (e.g., M_S or W_I_model).
    inputs : Array or Tuple
        The input for the model. 
        - For Naive: X_test (N, 6)
        - For PANN: (F_test, I_test) Tuple
    P_true : Array (N, 3, 3) or (N, 9)
        The ground truth stresses.
    W_true : Array (N,), optional
        The ground truth energies. If None, the energy plot is skipped.
    history : History Object, optional
        The history object from klax. If provided, the training loss is plotted.
    title : str
        Title prefix for the plots.
    """
    
    print(f"\n{'='*10} START EVALUATION: {title} {'='*10}")

    # --- 1. Generate Predictions ---
    # Use vmap to enable batch processing for single-sample models
    preds = jax.vmap(model)(inputs)

    # Distinguish: Does the model return only P (Naive) or (W, P) (PANN)?
    if isinstance(preds, tuple) and len(preds) == 2:
        # Case: PANN returns (W, P)
        W_pred, P_pred = preds
    else:
        # Case: Naive returns only P
        W_pred = None
        P_pred = preds

    # --- 2. Adjust Data Formats (Reshape to N, 9) ---
    # Ensure P_true and P_pred are flat arrays of shape (N, 9)
    if P_pred.ndim == 3: # (N, 3, 3) -> (N, 9)
        P_pred = P_pred.reshape(P_pred.shape[0], 9)
    
    if P_true.ndim == 3: # (N, 3, 3) -> (N, 9)
        P_true = P_true.reshape(P_true.shape[0], 9)

    # --- 3. Plot: Training History (Optional) ---
    if history is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(history.loss, linewidth=2, label="Training Loss")
        plt.yscale("log")
        plt.title(f"{title}: Training History")
        plt.xlabel("Step")
        plt.ylabel("Loss (Log)")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- 4. Plot: Energy Comparison (Optional) ---
    # Only plot if W_true is provided AND the model actually predicts W
    if W_true is not None and W_pred is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(W_true, 'k-', label="Ground Truth", linewidth=2, alpha=0.8)
        plt.plot(W_pred, 'r--', label="Prediction", linewidth=2)
        plt.title(f"{title}: Strain Energy Density (W)")
        plt.xlabel("Sample Index")
        plt.ylabel("W")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    elif W_true is not None and W_pred is None:
        print("Note: W_true was provided, but the model does not predict W (plot skipped).")

    # --- 5. Plot: Stress Evaluation ---
    # Call the existing subroutine for detailed stress metrics
    print(f"--- Stress Evaluation ({title}) ---")
    metrics = evaluate_MS_predictions(P_true, P_pred, title_prefix=title)
    
    return metrics

def plot_stress_stretch_comparison(pred_dict, F_true, P_true, component_indices=(0,0), title="Stress-Stretch Curve"):
    """
    Plots Stress (P) vs. Deformation (F) for a specific component.
    Crucial for checking physical validity (monotonicity, convexity).

    Parameters
    ----------
    pred_dict : dict
        Dictionary containing predictions. 
        Format: { "Label": (W_pred, P_pred) } or { "Label": P_pred }
    F_true : Array (N, 3, 3)
        Input deformation gradients (used for x-axis).
    P_true : Array (N, 3, 3) or (N, 9)
        Ground truth stress.
    component_indices : tuple (i, j)
        Indices of the component to plot (e.g., (0,0) for 11-component).
    title : str
        Plot title.
    """
    
    i, j = component_indices
    
    # --- 1. Prepare X-Axis (Deformation) ---
    # Extract the relevant component from F
    x_raw = F_true[:, i, j]
    
    # Determine sorted indices to prevent "spaghetti plots" if data is unsorted
    sort_idx = np.argsort(x_raw)
    x_sorted = x_raw[sort_idx]
    
    plt.figure(figsize=(10, 6))
    
    # --- 2. Plot Ground Truth ---
    # Ensure P_true is shaped (N, 3, 3) for indexing
    if P_true.ndim == 2:
        P_true_Reshaped = P_true.reshape(-1, 3, 3)
    else:
        P_true_Reshaped = P_true
        
    y_true = P_true_Reshaped[:, i, j][sort_idx]
    
    plt.plot(x_sorted, y_true, 'k-', linewidth=3, alpha=0.5, label="Ground Truth")
    
    # --- 3. Plot Models from Dictionary ---
    for label, preds in pred_dict.items():
        # Handle tuple (W, P) vs array P
        if isinstance(preds, tuple) and len(preds) == 2:
            _, P_pred = preds
        else:
            P_pred = preds
            
        # Reshape if necessary
        if P_pred.ndim == 2:
            P_pred = P_pred.reshape(-1, 3, 3)
            
        # Extract and sort y-values
        y_pred = P_pred[:, i, j][sort_idx]
        
        plt.plot(x_sorted, y_pred, '--', linewidth=2, label=label)

    # --- 4. Styling ---
    # Use 1-based indexing for label (11 instead of 00)
    comp_label = f"{i+1}{j+1}"
    plt.title(f"{title}\nComponent P_{{{comp_label}}} vs F_{{{comp_label}}}")
    plt.xlabel(f"Deformation Gradient F_{{{comp_label}}}")
    plt.ylabel(f"Piola-Kirchhoff Stress P_{{{comp_label}}}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


    """
    Erzeugt einen 'Textbuch-Style' Plot mit Plotly:
    - Achsen mit Pfeilen
    - Griechische Symbole (Sigma, Eta/Lambda)
    - Kein Kasten
    """
    
    i, j = component_indices
    
    # --- 1. Daten vorbereiten & sortieren ---
    # x-Achse: Deformation F (z.B. F_22)
    x_raw = F_true[:, i, j]
    
    # Sortieren ist essenziell für Linien-Plots!
    sort_idx = np.argsort(x_raw)
    x_sorted = x_raw[sort_idx]
    
    # Ground Truth sortieren
    # P sicherstellen als (N, 3, 3)
    if P_true.ndim == 2:
        P_true = P_true.reshape(-1, 3, 3)
    y_true = P_true[:, i, j][sort_idx]
    
    # --- 2. Plotly Figure erstellen ---
    fig = go.Figure()

    # Ground Truth (Dicke graue Linie)
    fig.add_trace(go.Scatter(
        x=x_sorted, 
        y=y_true,
        mode='lines',
        name='Ground Truth',
        line=dict(color='gray', width=4)
    ))

    # --- 3. Modelle hinzufügen ---
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blau, Orange, Grün
    styles = ['dash', 'dot', 'dashdot']
    
    for k, (label, preds) in enumerate(pred_dict.items()):
        # P extrahieren
        if isinstance(preds, tuple) and len(preds) == 2:
            _, P_pred = preds
        else:
            P_pred = preds
            
        if P_pred.ndim == 2:
            P_pred = P_pred.reshape(-1, 3, 3)
            
        y_pred = P_pred[:, i, j][sort_idx]
        
        fig.add_trace(go.Scatter(
            x=x_sorted, 
            y=y_pred,
            mode='lines',
            name=label,
            line=dict(color=colors[k % len(colors)], width=3, dash=styles[k % len(styles)])
        ))

    # --- 4. Styling (Der "Coole" Part) ---
    
    # Achsen-Limits bestimmen für Pfeil-Positionierung
    x_min, x_max = x_sorted.min(), x_sorted.max()
    y_min, y_max = min(y_true.min(), 0), y_true.max() # Y-Start meist bei 0 oder min
    
    # Buffer hinzufügen, damit die Pfeile Platz haben
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_limit = x_max + 0.05 * x_range
    y_limit = y_max + 0.1 * y_range

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, y=0.95),
        font=dict(family="Arial", size=14, color="black"),
        
        # Legend oben links "schwebend"
        legend=dict(
            x=0.05, y=0.95,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="Black",
            borderwidth=0
        ),
        plot_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=60, r=40, t=60, b=60), # Platz für Achsenbeschriftung
    )

    # ACHSEN KONFIGURATION (Dicke schwarze Linien, kein Rahmen)
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='lightgray',
        zeroline=False, # Wir bauen unsere eigene Linie
        showline=True, linewidth=3, linecolor='black', mirror=False, # mirror=False -> Kein Rahmen oben/rechts
        range=[x_min - 0.02*x_range, x_limit],
        title=dict(text=r"$\eta \text{ (Stretch in %)}$", font=dict(size=18)) # Latex Label
    )
    
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='lightgray',
        zeroline=False,
        showline=True, linewidth=3, linecolor='black', mirror=False,
        range=[y_min - 0.05*y_range, y_limit],
        title=dict(text=r"$\sigma \text{ (Stress)}$", font=dict(size=18)) # Latex Label
    )

    # --- 5. PFEILE HINZUFÜGEN (Annotations) ---
    # X-Achsen Pfeil
    fig.add_annotation(
        x=x_limit, y=y_min, # Ende der Achse
        ax=x_limit - 0.01 * x_range, ay=y_min, # Startpunkt des Pfeilkopfes (kurz davor)
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor="black"
    )
    # Y-Achsen Pfeil
    fig.add_annotation(
        x=x_min, y=y_limit, 
        ax=x_min, ay=y_limit - 0.01 * y_range,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor="black"
    )

    fig.show()

def plt_growth_cond(results, model_name="Model"):
    """
    Plot growth condition evaluation results.
    
    Parameters:
        results     : list of tuples (F, W) where 
                        F is a (3,3) deformation gradient
                        W is the predicted energy scalar
        model_name  : name of the model (for the plot title)
    """
    
    # Extract Frobenius norms and energies
    F_norms = []
    W_vals = []
    
    for F, W in results:
        frob = jnp.linalg.norm(F)          # Frobenius norm
        F_norms.append(float(frob))
        W_vals.append(float(W))
    
    # Convert to arrays for consistent plotting
    F_norms = jnp.array(F_norms)
    W_vals = jnp.array(W_vals)
    
    # Sort by X for smooth-looking curves
    idx = jnp.argsort(F_norms)
    F_norms = F_norms[idx]
    W_vals = W_vals[idx]
    
    # Create plot
    plt.figure(figsize=(7,5))
    plt.plot(F_norms, W_vals, "o-", markersize=4, label=model_name)
    
    plt.xlabel(r"$\|F\|_F$ (Frobenius norm of deformation gradient)")
    plt.ylabel(r"$W(F)$ (Predicted Energy)")
    plt.title(f"Growth Condition Evaluation - {model_name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
