"""Implementations of objective functions."""

import jax
import jax.numpy as jnp
import klax



# ----- Basic MSE used for standard non-sobolev NN training (unchanged) ----- #
"""Tasks: Used in all tasks"""


class MSE(klax.Loss):
    """Reimplementation of `klax.MSE` loss function."""

    def __call__(self, model, batch, batch_axis):
        x, y = batch
        y_pred = jax.vmap(model)(x)
        return jnp.mean(jnp.square(y - y_pred))



# ----- Custom sobolev loss class enabeling function data, gradient based or mixed training ----- #
"""Tasks: 2.3, 2.4"""


class SobolevLoss(klax.Loss):
    """
    A custom loss function for Sobolev training that balances
    loss on the function value and its gradient.
    
    Loss = alpha * Loss_f + beta * Loss_grad
    """
    alpha: float
    beta: float
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, model, batch, batch_axis):
        x, (y_true, y_grad_true) = batch
        
        y_pred, y_grad_pred = jax.vmap(model)(x)
        
        # --- MODIFICATION ---
        # y_pred has shape (N,) because output_dim="scalar"
        # y_true has shape (N, 1) from data.py
        # We must squeeze y_true to match y_pred.
        loss_f2 = jnp.mean((y_pred - y_true.squeeze())**2)
        # --- END MODIFICATION ---
        
        loss_grad = jnp.mean(jnp.sum((y_grad_pred - y_grad_true)**2, axis=-1))
        
        return self.alpha * loss_f2 + self.beta * loss_grad
    

class WeightedSobolevLoss(klax.Loss):
    """
    Weighted Sobolev loss.

    Expects batches of the form:
        batch = (x, ((W_true, P_true), w))

    where
        x       : inputs (e.g. (F, I) tuple for the PANN)
        W_true  : true energies, shape (N, 1) or (N,)
        P_true  : true stresses/gradients, shape (N, 3, 3) or (N, d_grad)
        w       : per-sample weights, shape (N,) or (N, 1) or (1, N)

    The loss is:
        L = alpha * mean(w_i * ||W_pred_i - W_true_i||^2)
          + beta  * mean(w_i * ||P_pred_i - P_true_i||^2)
    """

    alpha: float
    beta: float

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, model, batch, batch_axis):
        # Unpack batch: ((F, I), ((W_true, P_true), w))
        x, ((W_true, P_true), w) = batch

        # Forward pass: model returns (W_pred, P_pred)
        W_pred, P_pred = jax.vmap(model)(x)

        # Ensure W_true is 1D like W_pred
        W_true = W_true.squeeze()     # (N,)
        # Ensure weights are 1D length-N
        w = w.reshape(-1)             # (N,)

        # ---- Value (energy) term per sample ----
        # W_pred: (N,), W_true: (N,)
        err_W = (W_pred - W_true) ** 2          # (N,)

        # ---- Gradient / stress term per sample ----
        # IMPORTANT: use the same reduction as in your original SobolevLoss.
        # For P_pred, P_true shape (N, 3, 3):
        err_P = jnp.sum((P_pred - P_true) ** 2, axis=(1, 2))  # (N,)

        # Weighted means (consistent with WeightedMSE: mean(w * loss_i))
        loss_val  = jnp.mean(w * err_W)
        loss_grad = jnp.mean(w * err_P)

        return self.alpha * loss_val + self.beta * loss_grad




class WeightedMSE(klax.Loss):

    def __call__(self, model, batch, batch_axis):
        x, (y, w) = batch
        
        y_pred = jax.vmap(model)(x)
        w = w.squeeze()

        per_sample_loss = jnp.sum((y_pred - y)**2, axis=-1)
        return jnp.mean(w * per_sample_loss)
