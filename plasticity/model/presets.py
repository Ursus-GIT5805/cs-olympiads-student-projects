import jax
from model import *
from linear import linear, feedforward_linear, linears_from_array

def Resnet1_mnist(key):
    k1,k2,k3,k4 = jax.random.split(key, 4)

    params = [
        linear(784, 300, k1),
        linear(300, 300, k2),
        linear(300, 300, k3),
        linear(300, 10, k4),
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





def Resnet2_mnist(key):
    @jax.jit
    def resblock(params, a):
        p1, p2 = params

        x = feedforward_linear(p1, a)
        x = batch_norm(x)
        x = jax.nn.relu(x)

        x = feedforward_linear(p1, x)
        x = batch_norm(x)
        x = jax.nn.relu(a + x)
        return x

    params = [
        linear(784, 128, key),
        (linear(128, 128, key), linear(128, 128, key)),
        linear(128, 64, key),
        (linear(64, 64, key), linear(64, 64, key)),
        linear(64, 10, key),
    ]

    def run(params, a):
        a = feedforward_linear(params[0], a)

        a = resblock(params[1], a)
        a = jax.nn.relu(a)

        a = feedforward_linear(params[2], a)
        a = jax.nn.sigmoid( a )

        a = resblock(params[3], a)
        a = jax.nn.relu(a)

        a = feedforward_linear(params[4], a)
        a = jax.nn.softmax( a )
        return a

    return Model.init(
        params,
        jax.jit(run),
        input_dim=784,
        output_dim=10,
    )


def Linear1_mnist(key):
    params = linears_from_array([784, 100, 100, 10])
    def run(params, a):
        for p in params:
            a = feedforward_linear(p, a)
            a = jax.nn.sigmoid(a)
        return a

    return Model.init(
        params,
        jax.jit(run),
        input_dim=784,
        output_dim=10,
    )
