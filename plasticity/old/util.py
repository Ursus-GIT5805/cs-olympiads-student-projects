import jax
import jax.numpy as jnp
from dataclasses import dataclass

@jax.jit
def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

@jax.jit
def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1-sig)

@jax.tree_util.register_dataclass
@dataclass
class SquaredMeanCost:
    @staticmethod
    def fn(a, y):
        return 0.5*jnp.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)

@jax.tree_util.register_dataclass
@dataclass
class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        return jnp.sum(jnp.nan_to_num(y * jnp.log(a) + (1-y) * jnp.log1p(-a)))

    @staticmethod
    def delta(_z, a, y):
        return a - y
