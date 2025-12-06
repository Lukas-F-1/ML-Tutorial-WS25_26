import jax as jax
import jax.numpy as jnp
import numpy as np

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
    Berechnet die analytische Dehnungsenergie W basierend auf den Invarianten.
    
    Formel: W = 8*I1 + 10*J^2 - 56*log(J) + 0.2*(I4^2 + I5^2) - 44
    
    Parameters
    ----------
    I : ndarray, shape (N, 4)
        Array der Invarianten. 
        Erwartete Spaltenreihenfolge: [I1, J, I4, I5]
        
    Returns
    -------
    W : ndarray, shape (N,)
        Die berechnete Energie für jeden Zeitschritt.
    """

    I1 = I[:, 0]
    J  = I[:, 1]
    I4 = I[:, 2]
    I5 = I[:, 3]
    
    term_iso   = 8.0 * I1
    term_vol   = 10.0 * J**2 - 56.0 * jnp.log(J)
    term_aniso = 0.2 * (I4**2 + I5**2)
    const      = -44.0

    W = term_iso + term_vol + term_aniso + const
    
    return W

def compute_W_single(F, G_ti):
    """
    Computes the strain energy W for a single deformation gradient F.
    
    Parameters
    ----------
    F : array, shape (3, 3)
        Single deformation gradient
    G_ti : array, shape (3, 3)
        Transversely isotropic structural tensor
        
    Returns
    -------
    W : scalar
        Strain energy density
    """
    # Compute invariants for single F (reuse existing functions!)
    invariants = compute_all_invariants(F[None, :, :], G_ti)  # Add batch dim
    
    # Use existing analytical W function
    W = compute_analytical_W(invariants)
    return W[0]  # Remove batch dim

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

