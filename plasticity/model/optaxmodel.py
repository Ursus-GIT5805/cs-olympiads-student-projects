import jax
import jax.numpy as jnp
import optax

from dataclasses import dataclass
from functools import partial

# Example of a resnet block
@jax.jit
def resnet_block(a, w1, b1, w2, b2):
    # First transformation
    z1 = jnp.dot(a, w1) + b1
    h1 = jax.nn.relu(z1)
    # Second transformation
    z2 = jnp.dot(h1, w2) + b2
    # Skip connection
    return a + z2

@jax.jit
def forward(params, a):
    for w, b in params[:-1]:
        z = jnp.dot(a, w) + b
        a = jax.nn.relu(z)
    w, b = params[-1]
    z = jnp.dot(a, w) + b
    a = jax.nn.log_softmax(z)
    return a, z

@jax.jit
def crossentropy_cost(a, y, z):
    return -jnp.mean(jnp.sum(y * a, axis=1))

def gen_loss_function(run, cost):
    def loss_fn(params, x, y):
        a, z = forward(params, x)
        return cost(a, y, z)

    return jax.jit(loss_fn)

@partial(jax.jit, static_argnames=('optimizer', 'loss_fn'))
def train_step(params, opt_state, x, y, optimizer, loss_fn):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# =====

def init_layer_params(input_dim, output_dim, key):
    w_key, b_key = jax.random.split(key)
    weight = jax.random.normal(w_key, (input_dim, output_dim)) / jnp.sqrt(input_dim)
    bias = jax.random.normal(b_key, (output_dim,))
    return weight, bias

@jax.tree_util.register_dataclass
@dataclass
class Model:
    num_layers: int
    params: object
    cost: object # Cost function
    # activation: object # function

    @staticmethod
    def init(arr):
        key = jax.random.PRNGKey(len(arr))

        keys = jax.random.split(key, len(arr) - 1)
        params = [init_layer_params(m, n, k) for m, n, k in zip(arr[:-1], arr[1:], keys)]

        return Model(
            num_layers=len(arr),
            params=params,
            cost=crossentropy_cost,
        )

    def run(self, a):
        return forward(self.params, a)

    def train(
        self,
        train_x,
        train_y,
        epochs=10,
        batch_size=128,
        eta=0.01,
        return_score=False
    ):
        n = train_x.shape[0]
        optimizer = optax.sgd(learning_rate=eta)
        opt_state = optimizer.init(self.params)

        scores = []

        loss_fn = gen_loss_function(forward, self.cost)

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch+1, epochs))

            perm = jax.random.permutation(jax.random.PRNGKey(epoch), n)
            train_x, train_y = train_x[perm], train_y[perm]

            for i in range(0, n, batch_size):
                tx, ty = train_x[i : i+batch_size], train_y[i : i+batch_size]

                self.params, opt_state, loss = train_step(
                    self.params,
                    opt_state,
                    tx,
                    ty,
                    optimizer,
                    loss_fn,
                )

                if return_score:
                    scores.append(loss)

        if return_score:
            return scores

    def accuracy(
        self,
        test_x,
        test_y,
    ):
        a, _ = self.run(test_x)

        a_label = jnp.argmax(a, axis=1)
        t_label = jnp.argmax(test_y, axis=1)

        return jnp.sum(a_label == t_label) / test_x.shape[0] * 100

import matplotlib.pyplot as plt

def plot_scores(scores):
    plt.plot(scores)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    import loader

    model = Model.init([784, 200, 100, 10])
    train_data, test_data = loader.load_mnist_raw()

    train_x, train_y = train_data

    scores = model.train(train_x, train_y, epochs=10, batch_size=10, return_score=True)
    # plot_scores(scores)

    test_x, test_y = test_data

    acc = model.accuracy(test_x, test_y)
    print("Accuracy", acc)
