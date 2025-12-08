import jax as jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import os
import pandas as pd

def load_hyperelastic_data(filepath):
  """
  Loads data from a text file where each row contains:
  F11..F33 (9 numbers), P11..P33 (9 numbers), W (1 number).
  Returns arrays F, P, W where:
    F[i] is a 3x3 deformation gradient matrix
    P[i] is a 3x3 first Piola stress matrix
    W[i] is a scalar
  """

  # Load raw text data (N rows × 19 columns)
  raw = np.loadtxt(filepath)

  # Split into components
  F_raw = raw[:, 0:9]      # first 9 columns
  P_raw = raw[:, 9:18]     # next 9 columns
  W = raw[:, 18]           # last column

  # Reshape each row into a 3×3 matrix
  F = F_raw.reshape(-1, 3, 3)
  P = P_raw.reshape(-1, 3, 3)

  return F, P, W

def load_invariants(filepath):
  """
  Lädt die Invarianten aus einer Textdatei.
  Erwartetes Format: 4 Spalten (I1, J, I4, I5).
  
  Returns
  -------
  I : ndarray, shape (N, 4)
      Ein Array, das alle Invarianten enthält.
      Zugriff über Indizes:
      I[:, 0] -> I1
      I[:, 1] -> J
      I[:, 2] -> I4
      I[:, 3] -> I5
  """
  # Lade die rohen Daten
  raw = np.loadtxt(filepath)
  
  # Sicherheitscheck: Hat die Datei wirklich 4 Spalten?
  if raw.ndim == 1:
      # Falls nur eine Zeile existiert, reshape nötig
      raw = raw.reshape(1, -1)
      
  if raw.shape[1] != 4:
      raise ValueError(f"Erwarte 4 Spalten (I1, J, I4, I5), aber Datei hat {raw.shape[1]}")

  return raw

#helper for invariant computation
def cofactor(C):
    """
    Computes cof(C) = det(C) * inv(C)
    C has shape (...,3,3)
    """
    detC = jnp.linalg.det(C)
    C_inv = jnp.linalg.inv(C)
    cofC = detC[..., None, None] * C_inv
    return cofC

#all needed invariants
def compute_I1(F):
  """
  Computes I1 = tr(C) = tr(F^T F)
  
  Parameters
  ----------
  F : array, shape (..., 3, 3)
      Deformation gradient
      
  Returns
  -------
  I1 : array, shape (...)
      First invariant
  """
  C = jnp.swapaxes(F, -2, -1) @ F  # C = F^T F
  return jnp.trace(C, axis1=-2, axis2=-1)

def compute_I2(F):
    """
    I2 = tr(cof(C))
    """
    C = F.swapaxes(-1,-2) @ F
    cofC = cofactor(C)
    return jnp.trace(cofC, axis1=-2, axis2=-1)

def compute_J(F):
  """
  Computes J = det(F)
  
  Parameters
  ----------
  F : array, shape (..., 3, 3)
      Deformation gradient
      
  Returns
  -------
  J : array, shape (...)
      Determinant of deformation gradient
  """
  return jnp.linalg.det(F)

def compute_I4(F, G_ti):
  """
  Computes I4 = tr(C * G_ti) with C = F^T F
  
  Parameters
  ----------
  F : array, shape (..., 3, 3)
      Deformation gradient
  G_ti : array, shape (3, 3)
      Transversely isotropic structural tensor
      
  Returns
  -------
  I4 : array, shape (...)
      Fourth invariant
  """
  C = jnp.swapaxes(F, -2, -1) @ F  # C = F^T F
  CG = C @ G_ti
  return jnp.trace(CG, axis1=-2, axis2=-1)

def compute_I5(F, G_ti):
    """
    Computes I5 = tr(cof(C) * G_ti) with cof(C) = I3 * C^(-T)
    
    Parameters
    ----------
    F : array, shape (..., 3, 3)
        Deformation gradient
    G_ti : array, shape (3, 3)
        Transversely isotropic structural tensor
        
    Returns
    -------
    I5 : array, shape (...)
        Fifth invariant
    """
    C = jnp.swapaxes(F, -2, -1) @ F  # C = F^T F
    
    # I3 = det(C) = det(F^T F) = det(F)^2 = J^2
    I3 = jnp.linalg.det(C)
    
    # cof(C) = I3 * C^(-T) = I3 * (C^T)^(-1) = I3 * C^(-1) (since C is symmetric)
    C_inv = jnp.linalg.inv(C)
    cof_C = I3[..., None, None] * C_inv
    
    # tr(cof(C) * G_ti)
    result = cof_C @ G_ti
    return jnp.trace(result, axis1=-2, axis2=-1)

def compute_I7(F, G_cub):
    # C = Fᵀ F, shape (...,3,3)
    C = F.swapaxes(-1, -2) @ F

    # contraction 1: B_{kl} = sum_{ij} G_{ijkl} * C_{ij}
    B = jnp.einsum("ijkl,...ij->...kl", G_cub, C)

    # contraction 2: I7 = sum_{kl} B_{kl} * C_{kl}
    I7 = jnp.einsum("...kl,...kl->...", B, C)

    return I7


def compute_I11(F, G_cub):
    C = F.swapaxes(-1, -2) @ F

    # determinant of C
    detC = jnp.linalg.det(C)

    # inverse of C
    Cinv = jnp.linalg.inv(C)

    # cof(C) = det(C) * C^{-1}
    cofC = detC[..., None, None] * Cinv

    B = jnp.einsum("ijkl,...ij->...kl", G_cub, cofC)
    I11 = jnp.einsum("...kl,...kl->...", B, cofC)

    return I11


def compute_all_invariants(F, G_ti):
    """
    Computes all invariants (I1, J, I4, I5) simultaneously.
    
    Parameters
    ----------
    F : array, shape (N, 3, 3)
        Deformation gradients
    G_ti : array, shape (3, 3)
        Transversely isotropic structural tensor
        
    Returns
    -------
    invariants : array, shape (N, 4)
        Array with [I1, J, I4, I5] for each deformation state
    """
    I1 = compute_I1(F)
    J = compute_J(F)
    minus_J = -J
    I4 = compute_I4(F, G_ti)
    I5 = compute_I5(F, G_ti)
    
    return jnp.stack([I1, J, minus_J, I4, I5], axis=-1)

def compute_analytical_W(I):
    """
    Computes the analytical strain energy W based on invariants.

    Expected invariants:
        Either [I1, J, I4, I5]      (shape: N×4)
        or     [I1, J, -J, I4, I5]  (shape: N×5)

    The energy model uses only: I1, J, I4, I5
    """

    # --- Handle both shapes gracefully ---
    if I.shape[1] == 4:
        # Old format: [I1, J, I4, I5]
        I1 = I[:, 0]
        J  = I[:, 1]
        I4 = I[:, 2]
        I5 = I[:, 3]

    elif I.shape[1] == 5:
        # New format: [I1, J, -J, I4, I5]
        I1 = I[:, 0]
        J  = I[:, 1]
        I4 = I[:, 3]
        I5 = I[:, 4]

    else:
        raise ValueError(
            f"Invalid number of invariants: expected 4 or 5, got {I.shape[1]}"
        )

    # --- Compute energy ---
    term_iso   = 8.0 * I1
    term_vol   = 10.0 * J**2 - 56.0 * jnp.log(J)
    term_aniso = 0.2 * (I4**2 + I5**2)
    const      = -44.0

    return term_iso + term_vol + term_aniso + const


def compute_W_single(F, G_ti):
    """
    Computes the strain energy W for a single deformation gradient F.
    """

    # Compute all invariants (returns 5 entries now)
    invariants = compute_all_invariants(F[None, :, :], G_ti)

    # Analytical energy function already handles new format
    W = compute_analytical_W(invariants)

    return W[0]


def compute_P_batch(F_batch, G_ti):
    """
    Computes P = ∂W/∂F for a batch of deformation gradients.
    
    Parameters
    ----------
    F_batch : array, shape (N, 3, 3)
        Batch of deformation gradients
    G_ti : array, shape (3, 3)
        Transversely isotropic structural tensor
        
    Returns
    -------
    P_batch : array, shape (N, 3, 3)
        Batch of first Piola-Kirchhoff stresses
    """
    # Create gradient function
    grad_W = jax.grad(compute_W_single, argnums=0)
    
    # Vectorize over batch dimension
    compute_P_vectorized = jax.vmap(grad_W, in_axes=(0, None))
    
    return compute_P_vectorized(F_batch, G_ti)

def compute_path_weight(P_path):
    # Frobenius norm of each stress tensor
    norms = jnp.linalg.norm(P_path, axis=(1,2))
    return jnp.mean(norms)      # this is w

def add_minus_J(I_raw):
    I1 = I_raw[:, 0]
    J  = I_raw[:, 1]
    I4 = I_raw[:, 2]
    I5 = I_raw[:, 3]

    return jnp.column_stack([I1, J, -J, I4, I5])

def load_all_concentric(base_path):
    folder = os.path.join(base_path, "concentric")
    files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".txt")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    all_F = []

    for fname in files:
        full_path = os.path.join(folder, fname)
        data = np.loadtxt(full_path)
        F = data.reshape(-1, 3, 3)
        all_F.append(jnp.array(F))

    return all_F

def split_concentric_paths(all_F, test_size=0.2, key=jr.PRNGKey(0)):
    """
    Splits the concentric dataset into training and testing sets 
    based on entire load paths (NOT individual samples).

    Args:
        all_F: list of deformation gradient paths, each of shape (N_i, 3, 3)
        test_size: float in [0,1], fraction of load paths used for testing
        key: PRNGKey for random sampling (JAX style)

    Returns:
        F_train: (N_train, 3, 3)
        F_test:  (N_test, 3, 3)
        train_idx: indices of paths used for training
        test_idx:  indices of paths used for testing
    """
    
    num_paths = len(all_F)
    num_test_paths = int(num_paths * test_size)

    # Random permutation of path indices
    perm = jr.permutation(key, jnp.arange(num_paths))

    test_idx = perm[:num_test_paths]
    train_idx = perm[num_test_paths:]

    # Collect F tensors
    F_train_list = [all_F[int(i)] for i in train_idx]
    F_test_list  = [all_F[int(i)] for i in test_idx]

    # Collapse all paths into a single dataset
    F_train = jnp.concatenate(F_train_list, axis=0)  # shape (N_train, 3,3)
    F_test  = jnp.concatenate(F_test_list, axis=0)   # shape (N_test, 3,3)

    return F_train, F_test, train_idx, test_idx

def compute_concentric_dataset(F, G_ti):
    """
    Given deformation gradients F from the concentric dataset, compute:

        C(F), invariants I(F), W(F), P(F)

    using the existing helper functions in this file.

    Parameters
    ----------
    F : jnp.ndarray, shape (N, 3, 3)
        Deformation gradients (stacked)
    G_ti : jnp.ndarray, shape (3, 3)
        Structural tensor used for invariants I4, I5.

    Returns
    -------
    C : jnp.ndarray, shape (N, 3, 3)
        Right Cauchy–Green tensor C = FᵀF

    I : jnp.ndarray, shape (N, 5)
        Invariants [I1, J, -J, I4, I5]

    W : jnp.ndarray, shape (N,)
        Analytical strain energy

    P : jnp.ndarray, shape (N, 3, 3)
        Analytical stress tensor P = ∂W/∂F
    """

    # 1) Compute C = FᵀF
    C = jnp.einsum("nij,nkj->nik", F, F)  # fast and stable

    # 2) Compute invariants (already includes -J)
    I = compute_all_invariants(F, G_ti)

    # 3) Compute energy W
    # NOTE: compute_analytical_W expects shape (N,4) normally,
    #       but our compute_all_invariants now returns shape (N,5)
    #       with [I1, J, -J, I4, I5]
    # → analytical W uses only I1, J, I4, I5 → drop -J for W computation
    I_for_W = jnp.column_stack([I[:,0], I[:,1], I[:,3], I[:,4]])  
    W = compute_analytical_W(I_for_W)

    # 4) Compute analytical stresses
    P = compute_P_batch(F, G_ti)

    return C, I, W, P

import jax
import jax.numpy as jnp
import jax.random as jrandom


def preprocess_all_concentric(all_F, G_ti):
    """
    Precompute C, I, W, P and path weights for all loadpaths.
    
    Input:
        all_F: list of 100 arrays of shape (50,3,3)
        G_ti: structural tensor
    
    Returns:
        all_C, all_I, all_W, all_P, inv_path_weights
        (all as lists/arrays, same length as all_F)
    """
    all_C, all_I, all_W, all_P = [], [], [], []

    for F_path in all_F:
        C_path, I_path, W_path, P_path = compute_concentric_dataset(F_path, G_ti)
        all_C.append(C_path)
        all_I.append(I_path)
        all_W.append(W_path)
        all_P.append(P_path)

    # Compute path weights
    path_weights = jnp.array([compute_path_weight(P_path) for P_path in all_P])
    inv_path_weights = 1.0 / path_weights

    return all_C, all_I, all_W, all_P, inv_path_weights

def prepare_FFNN_split(
    all_C, all_P, inv_path_weights, 
    test_size=0.3,
    key=jrandom.PRNGKey(0)
):
    """
    Given precomputed per-path quantities, perform a new random train/test split
    and package an FFNN dataset.

    Inputs:
        all_C: list of path-wise C tensors  (100 × (50,3,3))
        all_P: list of path-wise P tensors  (100 × (50,3,3))
        inv_path_weights: array of shape (100,)
        test_size: fraction of paths to assign to test
        key: PRNGKey for random split

    Returns:
        train_data = (X_train, (Y_train, sample_weights))
        test_data  = (X_test,  Y_test)
        train_idx, test_idx
    """

    num_paths = len(all_C)
    num_test = int(num_paths * test_size)

    # --- Random split by path index ---
    perm = jrandom.permutation(key, jnp.arange(num_paths))
    test_idx = perm[:num_test]
    train_idx = perm[num_test:]

    # --- Flatten train/test sets ---
    C_train = jnp.concatenate([all_C[int(i)] for i in train_idx], axis=0)
    P_train = jnp.concatenate([all_P[int(i)] for i in train_idx], axis=0)

    C_test  = jnp.concatenate([all_C[int(i)] for i in test_idx], axis=0)
    P_test  = jnp.concatenate([all_P[int(i)] for i in test_idx], axis=0)

    # --- Build per-sample weights ---
    sample_weights = jnp.concatenate(
        [jnp.ones(all_P[int(i)].shape[0]) * inv_path_weights[int(i)]
         for i in train_idx],
        axis=0
    )

    # --- Convert to FFNN format ---
    X_train = jax.vmap(C_to_six)(C_train)
    X_test  = jax.vmap(C_to_six)(C_test)

    Y_train = P_train.reshape(len(P_train), 9)
    Y_test  = P_test.reshape(len(P_test), 9)

    train_data = (X_train, (Y_train, sample_weights))
    test_data  = (X_test, Y_test)

    return train_data, test_data, train_idx, test_idx

#helper fct for reduced parametrization of C because of symmetry
def C_to_six(C):
    return jnp.array([
        C[0,0], C[1,1], C[2,2],
        C[0,1], C[0,2], C[1,2],
    ])

def prepare_PANN_split(
    all_F, all_I, all_W, all_P,
    test_size=0.3,
    key=jax.random.PRNGKey(0)
):
    """
    Prepares a train/test split for the physics-augmented ICNN (W^I model).

    Inputs:
        all_F: list of deformation gradients per path (100 × (50,3,3))
        all_I: list of invariants per path (100 × (50,5))
        all_W: list of energies per path (100 × (50,))
        all_P: list of stresses per path (100 × (50,3,3))
        test_size: fraction of paths to assign to the test set
        key: PRNGKey for reproducible random split

    Returns:
        train_data = ((F_train, I_train), (W_train, P_train))
        test_data  = ((F_test,  I_test),  (W_test,  P_test))
        train_idx, test_idx  (path indices used)
    """

    num_paths = len(all_F)
    num_test = int(num_paths * test_size)

    # --- 1) Random split by path index ---
    perm = jax.random.permutation(key, jnp.arange(num_paths))
    test_idx = perm[:num_test]
    train_idx = perm[num_test:]

    # --- 2) Flatten train/test datasets ---
    F_train = jnp.concatenate([all_F[int(i)] for i in train_idx], axis=0)
    I_train = jnp.concatenate([all_I[int(i)] for i in train_idx], axis=0)
    W_train = jnp.concatenate([all_W[int(i)] for i in train_idx], axis=0)
    P_train = jnp.concatenate([all_P[int(i)] for i in train_idx], axis=0)

    F_test  = jnp.concatenate([all_F[int(i)] for i in test_idx], axis=0)
    I_test  = jnp.concatenate([all_I[int(i)] for i in test_idx], axis=0)
    W_test  = jnp.concatenate([all_W[int(i)] for i in test_idx], axis=0)
    P_test  = jnp.concatenate([all_P[int(i)] for i in test_idx], axis=0)

    # --- 3) Package datasets in the ICNN format ---
    train_data = ((F_train, I_train), (W_train, P_train))
    test_data  = ((F_test,  I_test),  (W_test,  P_test))

    return train_data, test_data, train_idx, test_idx


def load_multiscale_paths(path):
    """
    Loads CPShub multiscale dataset and returns a list of deformation paths.

    Returns
    -------
    F_paths : list of arrays, each (T_i, 3, 3)
    P_paths : list of arrays, each (T_i, 3, 3)
    W_paths : list of arrays, each (T_i,)
    J_paths : list of arrays, each (T_i,)
    mode_names : list of strings
    """
    store = pd.HDFStore(path, "r")
    keys = store.keys()

    F_paths = []
    P_paths = []
    W_paths = []
    J_paths = []
    mode_names = []

    for key in keys:
        df = store[key]

        # Extract data
        F = df[[f"F{i}{j}" for i in [1,2,3] for j in [1,2,3]]].values.reshape(-1, 3, 3)
        P = df[[f"P{i}{j}" for i in [1,2,3] for j in [1,2,3]]].values.reshape(-1, 3, 3)
        W = df["StrEn"].values
        J = df["J"].values

        F_paths.append(jnp.array(F))
        P_paths.append(jnp.array(P))
        W_paths.append(jnp.array(W))
        J_paths.append(jnp.array(J))
        mode_names.append(key)

    store.close()
    return F_paths, P_paths, W_paths, J_paths, mode_names

def G_cub():
    e1 = jnp.array([1.0, 0.0, 0.0])
    e2 = jnp.array([0.0, 1.0, 0.0])
    e3 = jnp.array([0.0, 0.0, 1.0])
    G = jnp.einsum("i,j,k,l->ijkl", e1, e1, e1, e1)
    G += jnp.einsum("i,j,k,l->ijkl", e2, e2, e2, e2)
    G += jnp.einsum("i,j,k,l->ijkl", e3, e3, e3, e3)
    return G


def compute_all_invariants_cubic(F, G_cub):
    """
    Computes invariants for cubic anisotropy:
    (I1, I2, J, -J, I7, I11)
    """
    C = F.swapaxes(-1,-2) @ F
    I1 = jnp.trace(C, axis1=-2, axis2=-1)

    # J invariants
    J = jnp.linalg.det(F)   # or sqrt(det(C))
    
    # I2 invariant
    I2 = compute_I2(F)

    # higher-order cubic invariants
    I7  = compute_I7(F, G_cub)
    I11 = compute_I11(F, G_cub)

    invariants = jnp.stack([I1, I2, J, -J, I7, I11], axis=-1)
    return invariants

