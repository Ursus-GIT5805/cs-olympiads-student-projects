import jax
import jax.numpy as jnp
import optax

from dataclasses import dataclass
from functools import partial

"""
# Example of a resnet block
@jax.jit
def resnet_block(a, w1, b1, w2, b2):
    z1 = jnp.dot(a, w1) + b1
    a = jax.nn.relu(z1)
    z2 = jnp.dot(a, w2) + b2
    return a + z2
"""
# =====

@jax.jit
def crossentropy_cost(a, y):
    return jnp.sum(jnp.nan_to_num(-y * jnp.log(a) - (1-y) * jnp.log1p(-a)))

# ===== Training =====

def gen_loss_function(run, cost):
    def loss_fn(params, x, y):
        a = run(params, x)
        return cost(a, y)

    return jax.jit(loss_fn)

@partial(jax.jit, static_argnames=('optimizer', 'loss_fn'))
def optimize(params, opt_state, x, y, optimizer, loss_fn):
    grads = jax.grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# ===== Model =====

@jax.tree_util.register_dataclass
@dataclass
class Model:
    input_dim: int
    output_dim: int
    params: object
    forward: object

    @staticmethod
    def init(
        params,
        forward,
        input_dim=None,
        output_dim=None,
    ):
        return Model(
            input_dim=input_dim,
            output_dim=output_dim,
            params=params,
            forward=forward,
        )

    def assert_data_shape(self, x, y):
        n = x.shape[0]

        if self.input_dim and train_x.shape != (n, self.input_dim):
            raise ValueError(
                "Input most be of shape {}, not {}"
                .format((n, self.input_dim), x.shape)
            )

        if self.output_dim and y.shape != (n, self.output_dim):
            raise ValueError(
                "Output most be of shape {}, not {}"
                .format(n, self.output_dim, y.shape)
            )

    def train(
        self,
        train_x,
        train_y,
        epochs=10,
        batch_size=128,
        optimizer=optax.sgd(learning_rate=0.01),
        cost=crossentropy_cost,
        evaluate=None # Return a list of losses per epoch
    ):
        n = train_x.shape[0]
        opt_state = optimizer.init(self.params)

        self.assert_data_shape(train_x, train_y)

        scores = []
        loss_fn = gen_loss_function(self.forward, cost)

        if evaluate:
            tx, ty = evaluate
            self.assert_data_shape(tx, ty)

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch+1, epochs))

            perm = jax.random.permutation(jax.random.PRNGKey(epoch), n)
            train_x, train_y = train_x[perm], train_y[perm]

            for i in range(0, n, batch_size):
                tx, ty = train_x[i : i+batch_size], train_y[i : i+batch_size]

                self.params, opt_state = optimize(
                    self.params,
                    opt_state,
                    tx,
                    ty,
                    optimizer,
                    loss_fn,
                )

            if evaluate:
                loss, _ = jax.value_and_grad(loss_fn)(self.params, tx, ty)
                scores.append(loss)
                print("Loss: {}".format(loss))

        if evaluate:
            return scores

    def accuracy(
        self,
        test_x,
        test_y,
    ):
        self.assert_data_shape(test_x, test_y)

        a = self.forward(self.params, test_x)

        a_label = jnp.argmax(a, axis=1)
        t_label = jnp.argmax(test_y, axis=1)

        return jnp.sum(a_label == t_label) / test_x.shape[0] * 100

if __name__ == "__main__":
    import loader
    from linear import *

    params = linears_from_array([784, 200, 100, 10])

    def run(params, a):
        for param in params:
            z = feedforward_linear(param, a)
            a = jax.nn.sigmoid(z)
        return a

    model = Model.init(params, jax.jit(run))
    train_data, test_data = loader.load_mnist_raw()

    train_x, train_y = train_data
    test_x, test_y = test_data

    scores = model.train(
        train_x, train_y,
        epochs=10, batch_size=10,
        # evaluate=(test_x[:1000], test_y[:1000])
    )

    acc = model.accuracy(test_x, test_y)
    print("Accuracy: {}%".format(acc))
