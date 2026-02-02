# load_model.py

from pathlib import Path
import equinox as eqx
import jax.random as jrandom
from typing import Callable, Any


def load_eqx_model(
    build_fn: Callable[..., Any],
    checkpoint_path: str | Path,
    *,
    key: jrandom.PRNGKey,
    **build_kwargs,
):
    """
    Load an Equinox model from a saved .eqx checkpoint.

    Parameters
    ----------
    build_fn : callable
        Function that builds the model architecture
        (e.g. build(), build_maxwell_nn(), build_gsm()).

    checkpoint_path : str or Path
        Path to the .eqx file.

    key : jax.random.PRNGKey
        PRNG key used to initialize the model structure.

    build_kwargs :
        All keyword arguments required to rebuild the model
        (e.g. hidden sizes, g, E_infty, etc.).

    Returns
    -------
    model :
        Loaded Equinox model ready for inference.
    """
    checkpoint_path = Path(checkpoint_path)

    # 1. Rebuild model structure
    model = build_fn(key=key, **build_kwargs)

    # 2. Load parameters
    model = eqx.tree_deserialise_leaves(checkpoint_path, model)

    return model
