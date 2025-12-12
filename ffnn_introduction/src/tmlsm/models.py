from typing import Callable, Sequence, Union

import equinox as eqx
import jax
from jax.nn.initializers import he_normal
from jaxtyping import PRNGKeyArray, Array
import jax.numpy as jnp
import klax
import optax
from . import losses as tl
from . import data_t2 as td2


# All model Classes we use:

# ----- General parametrized model class ----- #
"""Tasks: Used in all tasks"""


class Model(eqx.Module):
    """A custom trainable equinox.Module."""

    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activations: Sequence[Callable],
        *,
        key: PRNGKeyArray,
        constrain_icnn_weights: bool = False,
        fully_constrain_icnn_weights: bool = False
    ):
        """
        Initializes a feed-forward neural network.

        ICNN modes:
        - constrain_icnn_weights=True:
            Constrain all layers EXCEPT first (standard ICNN).

        - fully_constrain_icnn_weights=True:
            Constrain ALL layers INCLUDING first (needed for Task 3).

        Only one of these should be True at once.
        """

        # Safety check
        if constrain_icnn_weights and fully_constrain_icnn_weights:
            raise ValueError(
                "Choose either constrain_icnn_weights OR fully_constrain_icnn_weights, not both."
            )

        keys = jax.random.split(key, len(layer_sizes) - 1)
        built_layers = []

        # -------------------------
        # First layer
        # -------------------------
        first_wrapper = klax.NonNegative if fully_constrain_icnn_weights else None

        first_layer = klax.nn.Linear(
            layer_sizes[0],
            layer_sizes[1],
            weight_init=he_normal(),
            key=keys[0],
            weight_wrap=first_wrapper
        )
        built_layers.append(first_layer)

        # -------------------------
        # Remaining layers
        # -------------------------
        for in_size, out_size, k in zip(layer_sizes[1:-1], layer_sizes[2:], keys[1:]):

            if constrain_icnn_weights or fully_constrain_icnn_weights:
                wrapper = klax.NonNegative
            else:
                wrapper = None

            layer = klax.nn.Linear(
                in_size,
                out_size,
                weight_init=he_normal(),
                key=k,
                weight_wrap=wrapper
            )
            built_layers.append(layer)

        self.layers = tuple(built_layers)
        self.activations = tuple(activations)

    def __call__(self, x):
        """Performs the forward pass of the network."""
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x

# ----- SobolevModel Class ----- #
"""Tasks: 2.3 and 2.4"""


class SobolevModel(eqx.Module):
    """
    A wrapper model that computes both the value and the gradient
    of a contained network.
    """
    nn: Model  # This can be a standard Model or an ICNN

    def __init__(
        self,
        key: PRNGKeyArray,
        input_dim: int,
        output_dim: int,
        num_hidden_layers: int,
        nodes_per_layer: int,
        activation: Callable,
        is_icnn: bool,
        is_ficnn: bool,
    ):
        """
        Initializes the model, which internally builds the
        network it needs to differentiate.
        """
        key, nn_key = jax.random.split(key)
        
        self.nn = build(
            key=nn_key,
            input_dim=input_dim,
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,
            nodes_per_layer=nodes_per_layer,
            activations=activation,
            constrain_icnn_weights=is_icnn,
            fully_constrain_icnn_weights=is_ficnn
        )

    def __call__(self, x: Array) -> tuple[Array, Array]:
        """
        Takes a single input vector `x` (e.g., shape (2,)) and
        returns a tuple of (value, gradient).
        
        - value: The scalar output of the network (e.g., shape () or (1,)).
        - grad: The gradient of the value w.r.t. the input x
                (e.g., shape (2,)).
        
        Note: `klax.fit` will automatically `vmap` this.
        """
        # Get the scalar value
        value = self.nn(x)
        
        # Get the gradient function w.r.t. the input `x`
        grad_fn = jax.grad(self.nn)
        
        # Get the gradient vector
        grad = grad_fn(x)
        
        return value, grad
    
#for tasks 3, 4 (?)
class SobolevModel_WI_ti(eqx.Module):
    nn: eqx.Module
    G_ti: jnp.ndarray

    def __init__(
        self,
        G_ti,
        key: PRNGKeyArray,
        input_dim: int,
        output_dim: int,
        num_hidden_layers: int,
        nodes_per_layer: int,
        activation: Callable,
        is_icnn: bool,
        is_ficnn: bool
    ):
        
        """
        Initializes the model, which internally builds the
        network it needs to differentiate.
        """
        key, nn_key = jax.random.split(key)
        
        self.nn = build(        
        key=nn_key,
        input_dim=input_dim,
        output_dim=output_dim,
        num_hidden_layers=num_hidden_layers,
        nodes_per_layer=nodes_per_layer,
        activations=activation,
        constrain_icnn_weights=is_icnn,
        fully_constrain_icnn_weights=is_ficnn
        )
        self.G_ti = G_ti

    def compute_dI_dF(self, F):
        """Jacobian of invariants wrt a single F."""
        return jax.jacobian(
            lambda FF: td2.compute_all_invariants(FF[None, :, :], self.G_ti)[0],
            argnums=0
        )(F)

    def __call__(self, inputs):
        """Compute W and P for a SINGLE sample. klax.fit will vmap this."""
        
        F, I = inputs   # each of shape (3,3) and (5,)

        # Energy
        W = self.nn(I)

        # dW/dI
        dW_dI = jax.grad(self.nn)(I)

        # dI/dF
        dI_dF = self.compute_dI_dF(F)

        # Piola stress
        P = jnp.tensordot(dW_dI, dI_dF, axes=1)

        return W, P


#For task 5.2 using different invariants so jacobian computation uses a different fct and G is different
class SobolevModel_WI_Cubic(eqx.Module):
    nn: eqx.Module
    G_cub: jnp.ndarray

    def __init__(
        self,
        G_cub,
        key: PRNGKeyArray,
        input_dim: int,
        output_dim: int,
        num_hidden_layers: int,
        nodes_per_layer: int,
        activation: Callable,
        is_icnn: bool,
        is_ficnn: bool
    ):
        """
        Same structure as SobolevModel_WI but using cubic invariants.
        """
        key, nn_key = jax.random.split(key)

        # Neural network W(I)
        self.nn = build(
            key=nn_key,
            input_dim=input_dim,
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,
            nodes_per_layer=nodes_per_layer,
            activations=activation,
            constrain_icnn_weights=is_icnn,
            fully_constrain_icnn_weights=is_ficnn
        )

        self.G_cub = G_cub

    def compute_dI_dF(self, F):
        """
        Compute Jacobian of cubic invariants wrt F.
        Returns shape (6,3,3), corresponding to derivatives of 6 invariants.
        """
        return jax.jacobian(
            lambda FF: td2.compute_all_invariants_cubic(FF[None, :, :], self.G_cub)[0],
            argnums=0
        )(F)

    def __call__(self, inputs):
        """
        Compute W and P for a single sample.
        """
        F, I = inputs  # F: (3,3), I: (6,)

        # Compute W(I)
        W = self.nn(I)

        # dW/dI
        dW_dI = jax.grad(self.nn)(I)

        # dI/dF → shape (6,3,3)
        dI_dF = self.compute_dI_dF(F)

        # Piola stress P = sum_k dW/dI_k * dI_k/dF
        P = jnp.tensordot(dW_dI, dI_dF, axes=1)

        return W, P

# For task 5.3 Polyconvex ICNN as ICNN(F, cof F, det F)=W(F)
class SobolevModel_WF(eqx.Module):
    """
    Polyconvex deformation-gradient-based neural network model W^F.
    Inputs: y(F) = (F, cofF, detF) ∈ R^19
    Output: W(F), P(F)
    """
    nn: eqx.Module

    def __init__(
        self,
        key,
        input_dim=19,
        output_dim=1,
        num_hidden_layers=3,
        nodes_per_layer=64,
        activation=jax.nn.tanh,
        is_icnn=False,
        is_ficnn=False
    ):
        key, nn_key = jax.random.split(key)

        self.nn = build(
            key=nn_key,
            input_dim=input_dim,
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,
            nodes_per_layer=nodes_per_layer,
            activations=activation,
            constrain_icnn_weights=is_icnn,
            fully_constrain_icnn_weights=is_ficnn
        )

    def compute_inputs(self, F):
        detF = jnp.linalg.det(F)
        cofF = detF * jnp.linalg.inv(F).T
        return jnp.concatenate([F.reshape(-1),
                                cofF.reshape(-1),
                                detF.reshape(-1)])

    def compute_dinputs_dF(self, F):
        # Jacobian dy/dF, shape (19,3,3)
        return jax.jacobian(self.compute_inputs)(F)

    def __call__(self, inputs):
        """
        inputs: F (3×3)
        returns: (W, P)
        """
        F = inputs

        y = self.compute_inputs(F)           # (19,)
        W = self.nn(y)                       # scalar output

        dW_dy = jax.grad(self.nn)(y)         # (19,)

        dy_dF = self.compute_dinputs_dF(F)   # (19,3,3)

        # Contract: P_ij = sum_k dW/dy_k * dy_k/dF_ij
        P = jnp.tensordot(dW_dy, dy_dF, axes=1)   # (3,3)

        return W, P



# All helper functions for building and training models



# ----- General parametrized build function ----- #
"""Tasks: Used in all tasks in combination with Model or SobolevModel classes"""


def build(
    *,
    key: PRNGKeyArray,
    input_dim: int,
    output_dim: int,
    num_hidden_layers: int,
    nodes_per_layer: int,
    activations: Union[Callable, Sequence[Callable]],
    constrain_icnn_weights: bool = False,
    fully_constrain_icnn_weights: bool = False
):
    """
    Builds and returns a model instance with flexible activation functions.
    """
    layer_sizes = (
        [input_dim]
        + [nodes_per_layer] * num_hidden_layers
        + [output_dim]
    )

    # Check if a list of activations was provided or just one
    if isinstance(activations, (list, tuple)):
        expected_len = num_hidden_layers + 1
        if len(activations) != expected_len:
            raise ValueError(
                f"Received a list of {len(activations)} activation functions, "
                f"but expected {expected_len} for {num_hidden_layers} "
                "hidden layers plus an output layer."
            )
        final_activations = activations
    else:
        # If it's a single function, create the list with a linear output
        final_activations = [activations] * num_hidden_layers + [lambda x: x]

    return Model(
        layer_sizes=layer_sizes,
        activations=final_activations,
        key=key,
        constrain_icnn_weights=constrain_icnn_weights,
        fully_constrain_icnn_weights = fully_constrain_icnn_weights
    )

# ----- General parametrized train_model function ----- #
"""Tasks: Used in all tasks in combination with Model or SobolevModel classes"""
def train_model(
    model,
    train_data,
    key,
    steps,
    batch_size,
    learning_rate,
    loss_fn=None  # necessary for sobolev custom loss
):
    """Trains a single model instance and returns it with its history."""
    
    if loss_fn is None:
        loss_fn = tl.MSE()
        
    history_callback = klax.HistoryCallback(log_every=100, verbose=False)

    trained_model, history = klax.fit(
        model,
        train_data,
        batch_size=batch_size,
        steps=steps,
        loss_fn=loss_fn,
        optimizer=optax.adam(learning_rate),
        history=history_callback,
        key=key,
        #sample weight
    )
    return trained_model, history

def train_WI(
    model,
    train_data,
    key,
    steps,
    batch_size,
    learning_rate,
    loss_fn 
):
    """
    train_data = ((F_train, I_train), (W_true, P_true))
    """

    history = klax.HistoryCallback(log_every=100, verbose=False)

    trained_model, history = klax.fit(
        model,
        train_data,
        batch_size=batch_size,
        steps=steps,
        loss_fn=loss_fn,
        optimizer=optax.adam(learning_rate),
        history=history,
        key=key,
    )

    return trained_model, history
