import jax
from model import *

def FirstResnet_Mnist():
    params = [
        linear(784, 100, key),
        linear(100, 100, key),
        linear(100, 100, key),
        linear(100, 10, key),
    ]

    def run(params, a):
        a = feedforward_linear(params[0], a)

        x1 = a.copy()

        a = jax.nn.sigmoid(a)
        a = feedforward_linear(params[1], a)
        a = batch_norm(a)
        a = jax.nn.relu(a)

        a = feedforward_linear(params[2], a)
        a = batch_norm(a)

        a = a + x1
        a = jax.nn.relu(a)

        a = feedforward_linear(params[3], a)
        a = jax.nn.softmax( a )
        return a

    return Model.init(
        params,
        jax.jit(run),
        input_dim=784,
        output_dim=10,
    )
