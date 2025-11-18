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


