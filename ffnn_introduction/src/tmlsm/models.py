"""Model implementations."""

from typing import Callable, Sequence, Union

import equinox as eqx
import jax
from jax.nn.initializers import he_normal
from jaxtyping import PRNGKeyArray
import klax
import optax
from . import losses as tl


class Model(eqx.Module):
    """A custom trainable equinox.Module."""

    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activations: Sequence[Callable],
        *,
        key: PRNGKeyArray
    ):
        """
        Initializes a feed-forward neural network.
        """
        keys = jax.random.split(key, len(layer_sizes) - 1)

        self.layers = tuple(
            klax.nn.Linear(
                in_size,
                out_size,
                weight_init=he_normal(),
                key=k
            ) for in_size, out_size, k in zip(
                layer_sizes[:-1], layer_sizes[1:], keys
            )
        )
        self.activations = tuple(activations)

    def __call__(self, x):
        """Performs the forward pass of the network."""
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x


def build(
    *,
    key: PRNGKeyArray,
    input_dim: int,
    output_dim: int,
    num_hidden_layers: int,
    nodes_per_layer: int,
    activations: Union[Callable, Sequence[Callable]],
):
    """
    Builds and returns a model instance with flexible activation functions.

    Args:
        key: The JAX random key for weight initialization.
        input_dim: The dimension of the input vector.
        output_dim: The dimension of the output vector.
        num_hidden_layers: The number of hidden layers in the network.
        nodes_per_layer: The number of nodes (neurons) in each hidden layer.
        activations: Can be either a single activation function (which will be
                     applied to all hidden layers, with a linear output layer)
                     or a sequence (list/tuple) of activation functions. If a
                     sequence is provided, its length must be exactly
                     `num_hidden_layers + 1` (one for each hidden layer plus
                     one for the output layer).
    Returns:
        An instance of the Model class.
    """
    layer_sizes = (
        [input_dim]
        + [nodes_per_layer] * num_hidden_layers
        + [output_dim]
    )

    # Check if a list of activations was provided or just one
    if isinstance(activations, (list, tuple)):
        # If it's a list, check if the length is correct
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
        key=key
    )

# --- The Reusable Training Function (Unchanged) ---
def train_model(model, train_data, key, steps, batch_size, learning_rate):
    """Trains a single model instance and returns it with its history."""
    trained_model, history = klax.fit(
        model,
        train_data,
        batch_size=batch_size,
        steps=steps,
        loss_fn=tl.MSE(),
        optimizer=optax.adam(learning_rate),
        history=klax.HistoryCallback(log_every=1, verbose=False), # Log less frequently for cleaner output
        key=key,
    )
    return trained_model, history