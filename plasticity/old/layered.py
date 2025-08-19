import jax
import jax.numpy as jnp
from dataclasses import dataclass

from util import *

@jax.jit
def backprop1(
    model,
    data,
):
    acts = [data]
    zs = []

    num = 0
    for b, w in zip(model.biases, model.weights):
        z = jnp.dot(w, a) + b
        a = sigmoid(z)

        zs.append(z)
        acts.append(a)
        num += 1

    return (num, acts, zs), a

@jax.jit
def backprop2(
    model,
    data,
    delta,
):
    num, acts, zs = data

    err_b = [jnp.zeros(b.shape) for b in model.biases]
    err_w = [jnp.zeros(w.shape) for w in model.weights]

    err_b[-1] = delta
    err_w[-1] = jnp.dot(delta, activations[-2].transpose())

    for l in range(2, num):
        z = zs[-l]
        delta = jnp.dot(model.weights[-l+1].transpose(), delta) * sigmoid_prime(z)
        err_b[-l] = delta
        err_w[-l] = jnp.dot(delta, activations[-l-1].transpose())

    return (err_b, err_w), delta

@jax.jit
def feedforward(layered_model, a):
    for b, w in zip(layered_model.biases, layered_model.weights):
        z = jnp.matmul(w, a) + b
        a = sigmoid(z)
    return a

@jax.tree_util.register_dataclass
@dataclass
class Layered:
    biases: list[jnp.ndarray]
    weights: list[jnp.ndarray]
    cost: object

    def from_array(arr):
        num_layers = len(arr)
        keys_bias = jax.random.split(jax.random.PRNGKey(2), num_layers-1)

        biases = [
            jax.random.normal(k, shape=(shp,1))
            for k, shp in zip(keys_bias, arr[1:])
        ]

        keys_weights = jax.random.split(jax.random.PRNGKey(4), num_layers-1)
        shapes = zip(arr[1:], arr[:-1])

        weights = [
            jax.random.normal(k, shape=(x,y)) / jnp.sqrt(x)
            for k, (x,y) in zip(keys_weights, shapes)
        ]

        return Layered(
            biases=biases,
            weights=weights,
            cost=CrossEntropyCost(),
        )

    def run(self, a):
        return feedforward(self, a)

    def backprop1(self, data):
        return backprop1(self, data)

    def backprop2(self, data, delta=None, label=None):
        if delta == None and label == None:
            raise BaseException("Either delta or label must be given")

        _num, acts, zs = data
        if delta == None and label:
            delta = model.cost.delta(zs[-1], acts[-1], label)

        errs, delta = backprop2(model, data, delta)

        return delta

    def update_to_errs(self, errs, eta):
        err_b, err_w = errs

        self.biases = [
            b - (eta/sz) * db for b, db in zip(self.biases, err_b)
        ]
        self.weights = [
            w - (eta/sz) * dw for w, dw in zip(self.weights, err_w)
        ]
