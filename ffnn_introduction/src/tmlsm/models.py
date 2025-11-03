from typing import Callable, Sequence, Union

import equinox as eqx
import jax
from jax.nn.initializers import he_normal
from jaxtyping import PRNGKeyArray, Array
import klax
import optax
from . import losses as tl



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
        constrain_icnn_weights: bool = False
    ):
        """
        Initializes a feed-forward neural network.
        
        If 'constrain_icnn_weights' is True, applies non-negative
        weight constraints to all layers except the first.
        """
        keys = jax.random.split(key, len(layer_sizes) - 1)
        
        built_layers = []
        
        # --- Build the first layer (always unconstrained) ---
        first_layer = klax.nn.Linear(
            layer_sizes[0],
            layer_sizes[1],
            weight_init=he_normal(),
            key=keys[0],
            weight_wrap=None  # Explicitly unconstrained
        )
        built_layers.append(first_layer)
        
        # --- Build all subsequent layers (constrained or unconstrained) ---
        for in_size, out_size, k in zip(layer_sizes[1:-1], layer_sizes[2:], keys[1:]):
            
            # Conditionally set the wrapper
            wrapper = klax.NonNegative if constrain_icnn_weights else None
            
            layer = klax.nn.Linear(
                in_size,
                out_size,
                weight_init=he_normal(),
                key=k,
                weight_wrap=wrapper  # <-- This is the new, correct way
            )
            built_layers.append(layer)

        self.layers = tuple(built_layers)
        self.activations = tuple(activations)

    def __call__(self, x):
        """Performs the forward pass of the network."""
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x



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
    constrain_icnn_weights: bool = False
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
        constrain_icnn_weights=constrain_icnn_weights
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
    )
    return trained_model, history



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
        is_icnn: bool
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
            constrain_icnn_weights=is_icnn
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