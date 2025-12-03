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
    
    
class WeightedMSE(klax.Loss):
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, model, batch, batch_axis):
        x, (y, w) = batch
        
        y_pred = jax.vmap(model)(x)
        w = w.squeeze()

        per_sample_loss = jnp.sum((y_pred - y)**2, axis=-1)
        return jnp.mean(w * per_sample_loss)
