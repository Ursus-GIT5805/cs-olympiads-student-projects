import jax
import jax.numpy as jnp
import optax
import random
import dill as pickle

from dataclasses import dataclass
from functools import partial

@jax.jit
def batch_norm(x):
    mean = jnp.mean(x)
    var = jnp.var(x)
    return jnp.nan_to_num((x-mean) / jnp.sqrt(var))

# =====
def _tree_random_keys(key, pytree):
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    keys = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(treedef, keys)

def reset_weights_normal(params, key, p=0.2):
    """
    Re-init each WEIGHT element with prob p to N(0, 1/sqrt(in_features)).
    Biases (1D arrays) remain unchanged.
    """
    key_tree = _tree_random_keys(key, params)

    def reset_array(x, k):
        if x.ndim == 2:  # treat as weight matrix
            in_features = x.shape[0]
            scale = 1.0 / jnp.sqrt(in_features)
            k_mask, k_noise = jax.random.split(k)
            mask  = jax.random.bernoulli(k_mask, p, x.shape)
            x_new = jax.random.normal(k_noise, x.shape) * scale
            return jnp.where(mask, x_new, x)
        else:
            return x  # biases or other non-2D leaves

    return jax.tree_util.tree_map(reset_array, params, key_tree)

@jax.jit
def kl_divergence(p, q):
    eps=1e-12
    p = jnp.clip(p, eps, 1.0)
    q = jnp.clip(q, eps, 1.0)
    return jnp.mean(p * (jnp.log(p) - jnp.log(q)))

@jax.jit
def kl_divergence_cost(a, y):
    return kl_divergence(y, a)

# =====

@jax.jit
def crossentropy_cost(a, y):
    eps = 0.001
    return jnp.mean(-y * jnp.log(a+eps) - (1-y) * jnp.log1p(-a+eps))

@jax.jit
def squaredmean_cost(a, y):
    return jnp.mean( (a-y) ** 2 )

# ===== Training =====

def _gen_loss_function(
        run,
        cost,
        l2=False,
        l2_eps=1e-4
):
    if l2:
        def loss_fn(params, x, y):
            a = run(params, x)
            l2_loss = sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))
            return cost(a, y) + l2_loss*l2_eps

    else:
        def loss_fn(params, x, y):
            a = run(params, x)
            return cost(a, y)

    return jax.jit(loss_fn)

@partial(jax.jit, static_argnames=('optimizer', 'loss_fn', 'batches', 'batch_size'))
def train_epoch(params, opt_state, x, y, optimizer, loss_fn, batches, batch_size):

    def step(carry, batch_idx):
        params, opt_state = carry
        start = batch_idx * batch_size
        xb = jax.lax.dynamic_slice(x, (start, 0), (batch_size, x.shape[1]))
        yb = jax.lax.dynamic_slice(y, (start, 0), (batch_size, y.shape[1]))

        loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), jnp.arange(batches))
    return params, opt_state, losses



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

        if self.input_dim and (x.shape != (n, self.input_dim)):
            raise ValueError(
                "Input most be of shape {}, not {}"
                .format((n, self.input_dim), x.shape)
            )

        if self.output_dim and (y.shape != (n, self.output_dim)):
            raise ValueError(
                "Output most be of shape {}, not {}"
                .format(n, self.output_dim, y.shape)
            )
    # def resetsubset(self):
    #     newparams=[]
    #     for param in self.params:
    #         newparamweights=0
    #         for w1 in param[0]:
    #             for w2 in w1:
    #                 if(random.random()<0.2):
    
    def model_reset_subset(self, p=0.2, seed=0):
        key = jax.random.PRNGKey(seed)
        self.params = reset_weights_normal(self.params, key, p=p)

    def train(
        self,
        train_x,
        train_y,
        epochs=10,
        batch_size=128,
        optimizer=optax.sgd(learning_rate=0.01),
        cost=crossentropy_cost,
        return_score=False, # Returns a list of losses per batch
        opt_state=None,
        evaluate=None, # Prints a list of losses corresponding to the given test data
        seed=42,
        batches=None,
        verbose=True,
        l2=False,
        l2_eps=1e-4,
        eval_fn=None,
    ):
        n = train_x.shape[0]

        if opt_state == None:
            opt_state = optimizer.init(self.params)

        self.assert_data_shape(train_x, train_y)

        r = n
        if not batches:
            batches = train_x.shape[0] // batch_size
        if not batch_size:
            batch_size = train_x.shape[0] // batches

        scores = []
        loss_fn = _gen_loss_function(self.forward, cost, l2=l2, l2_eps=l2_eps)
        if not eval_fn:
            eval_fn = cost
        eval_fn = _gen_loss_function(self.forward, eval_fn, l2=l2, l2_eps=l2_eps)

        if evaluate:
            tx, ty = evaluate
            self.assert_data_shape(tx, ty)

        for epoch in range(epochs):
            if verbose:
                print("Epoch {}/{}".format(epoch+1, epochs))

            key = jax.random.PRNGKey(seed)
            train_x = jax.random.permutation(key, train_x, axis=0)
            train_y = jax.random.permutation(key, train_y, axis=0)

            self.params, opt_state, loss = train_epoch(
                params=self.params,
                opt_state=opt_state,
                x=train_x,
                y=train_y,
                optimizer=optimizer,
                loss_fn=loss_fn,
                batches=batches,
                batch_size=batch_size
            )

            if return_score:
                scores.append(jnp.mean(loss))

            if evaluate:
                loss, _ = jax.value_and_grad(eval_fn)(self.params, tx, ty)
                scores.append(loss)
                print("Loss: {}".format(loss))

        if return_score:
            return scores, opt_state

        return opt_state

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

    def evaluate(self, a):
        return self.forward(self.params, a)

    def save(self, path, overwrite=False):
        mode = "wb" if overwrite else "xb"

        with open(path, mode) as f:
            pickle.dump(self, f)

    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
