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
    I4 = compute_I4(F, G_ti)
    I5 = compute_I5(F, G_ti)
    
    return jnp.stack([I1, J, I4, I5], axis=-1)

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
