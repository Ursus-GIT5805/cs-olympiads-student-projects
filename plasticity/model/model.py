import jax
import jax.numpy as jnp
import optax

from dataclasses import dataclass
from functools import partial

@jax.jit
def batch_norm(x):
    mean = jnp.mean(x)
    var = jnp.var(x)
    return jnp.nan_to_num((x-mean) / jnp.sqrt(var))

# =====

@jax.jit
def crossentropy_cost(a, y):
    eps = 0.001
    return jnp.mean(-y * jnp.log(a+eps) - (1-y) * jnp.log1p(-a+eps))

@jax.jit
def squaredmean_cost(a, y):
    return jnp.mean( (a-y) ** 2 )

# ===== Training =====

def _gen_loss_function(run, cost):
    def loss_fn(params, x, y):
        a = run(params, x)
        return cost(a, y)

    return jax.jit(loss_fn)

@partial(jax.jit, static_argnames=('optimizer', 'loss_fn'))
def train_step(params, opt_state, x, y, optimizer, loss_fn):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# ===== Model =====

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

        if self.input_dim and (train_x.shape != (n, self.input_dim)):
            raise ValueError(
                "Input most be of shape {}, not {}"
                .format((n, self.input_dim), x.shape)
            )

        if self.output_dim and (y.shape != (n, self.output_dim)):
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
        return_score=False,
        evaluate=None, #     Return a list of losses per epoch
        seed=None,
        batches=1e6
    ):
        n = train_x.shape[0]
        opt_state = optimizer.init(self.params)

        self.assert_data_shape(train_x, train_y)

        scores = []
        loss_fn = _gen_loss_function(self.forward, cost)

        if evaluate:
            tx, ty = evaluate
            self.assert_data_shape(tx, ty)

        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch+1, epochs))

            perm = jax.random.permutation(jax.random.PRNGKey(seed if seed else epoch), n)
            train_x, train_y = train_x[perm], train_y[perm]

            for i in range(0, min(n,batch_size*batches), batch_size):
                tx, ty = train_x[i : i+batch_size], train_y[i : i+batch_size]

                self.params, opt_state, loss = train_step(
                    self.params,
                    opt_state,
                    tx,
                    ty,
                    optimizer,
                    loss_fn,
                )

                if return_score: scores.append(loss)

            if evaluate:
                loss, _ = jax.value_and_grad(loss_fn)(self.params, tx, ty)
                # scores.append(loss)
                print("Loss: {}".format(loss))

        if return_score:
            return scores

    def loss(
        self,
        tx,
        ty,
        cost=crossentropy_cost,
    ):
        loss_fn = _gen_loss_function(self.forward, cost)
        loss, _ = jax.value_and_grad(loss_fn)(self.params, tx, ty)
        return loss

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

    def evaluate(self, test_data):
        solution = []
        for x in test_data:
            a = self.forward(self.params,x)
            solution.append( a)
        jnpsol = jnp.array(solution)
        jnpsol.reshape(test_data.shape[0],10)
        return jnpsol
