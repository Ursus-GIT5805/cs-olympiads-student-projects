"""
Methods working with linear layers
"""

import jax
import jax.numpy as jnp

def linear(input_dim, output_dim, key):
    """
    Returns a tuple (weight, bias) of a simple Linear layer with.

    key: jax.random.key for randomization
    """
    w_key, b_key = jax.random.split(key)
    weight = jax.random.normal(w_key, (input_dim, output_dim)) / jnp.sqrt(input_dim)
    bias = jax.random.normal(b_key, (output_dim))

    return weight, bias


def linears_from_array(arr, key=None):
    """
    Create a list of linear neural layers, with 'arr' being the number of neurons per layer.
    """

    if key is None:
        key = jax.random.PRNGKey(len(arr))

    keys = jax.random.split(key, len(arr) - 1)
    params = [linear(m, n, k) for m, n, k in zip(arr[:-1], arr[1:], keys)]

    return params


@jax.jit
def feedforward_linear(params, a):
    """
    Feed forward the parameters of a linear layer.
    """
    w, b = params
    return jnp.dot(a, w) + b
