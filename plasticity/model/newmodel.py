import jax
import jax.numpy as jnp
import optax
import dill as pickle

from dataclasses import dataclass
from functools import partial
import math

# ========= Utils =========

@jax.jit
def batch_norm(x):
    mean = jnp.mean(x)
    var = jnp.var(x)
    return jnp.nan_to_num((x - mean) / jnp.sqrt(var))

@jax.jit
def kl_divergence(p, q):
    eps = 1e-12
    p = jnp.clip(p, eps, 1.0)
    q = jnp.clip(q, eps, 1.0)
    return jnp.mean(p * (jnp.log(p) - jnp.log(q)))

@jax.jit
def kl_divergence_cost(a, y):
    return kl_divergence(y, a)

@jax.jit
def crossentropy_cost(a, y):
    eps = 1e-3
    return jnp.mean(-y * jnp.log(a + eps) - (1 - y) * jnp.log1p(-(a - eps)))

@jax.jit
def squaredmean_cost(a, y):
    return jnp.mean((a - y) ** 2)


def _gen_loss_function(run, cost, l2=False, l2_eps=1e-4):
    if l2:
        def loss_fn(params, x, y):
            a = run(params, x)
            l2_loss = sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(params) if isinstance(p, jnp.ndarray))
            return cost(a, y) + l2_loss * l2_eps
    else:
        def loss_fn(params, x, y):
            a = run(params, x)
            return cost(a, y)

    return jax.jit(loss_fn)


# ========= Slow-down machinery =========

def _topk_threshold(vec_abs, p: float):
    n = vec_abs.size
    if n == 0:
        return None
    k = int(math.floor(p * n))
    if k <= 0:
        return None
    # kth-largest via partition (faster than full sort)
    idx = n - k
    return jnp.partition(vec_abs, idx)[idx]

def build_slow_mask(old_params, new_params, p=0.2, weights_only=False):
    """
    Returns a pytree of {0.0,1.0} masks: 1 on the top-p fraction by |Δ| (element-wise).
    If weights_only=True, only 2D leaves are considered; 1D leaves (biases) get 0.
    """
    diffs = jax.tree_util.tree_map(
        lambda a, b: jnp.abs(a - b) if (isinstance(a, jnp.ndarray) and isinstance(b, jnp.ndarray)) else a,
        new_params, old_params
    )

    leaves, treedef = jax.tree_util.tree_flatten(diffs)

    vecs = []
    kinds = []
    for d in leaves:
        if isinstance(d, jnp.ndarray):
            if weights_only and d.ndim != 2:
                kinds.append("skip")
            else:
                kinds.append("use")
                vecs.append(d.ravel())
        else:
            kinds.append("skip")

    if not vecs:
        return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) if isinstance(x, jnp.ndarray) else x, new_params)

    vec_abs = jnp.concatenate(vecs, axis=0)
    thr = _topk_threshold(vec_abs, p)
    if thr is None:
        return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) if isinstance(x, jnp.ndarray) else x, new_params)

    mask_leaves = []
    for d, kind in zip(leaves, kinds):
        if isinstance(d, jnp.ndarray) and kind == "use":
            mask = (d >= thr).astype(d.dtype)  # 1 where |Δ| >= thr
            mask_leaves.append(mask)
        elif isinstance(d, jnp.ndarray):
            mask_leaves.append(jnp.zeros_like(d))
        else:
            mask_leaves.append(d)

    return jax.tree_util.tree_unflatten(treedef, mask_leaves)


@partial(jax.jit, static_argnames=('optimizer', 'loss_fn', 'batches', 'batch_size'))
def train_epoch(params, opt_state, x, y, optimizer, loss_fn, batches, batch_size,
                slow_mask=None, gamma=0.2):

    def step(carry, batch_idx):
        params, opt_state = carry
        start = batch_idx * batch_size
        xb = jax.lax.dynamic_slice(x, (start, 0), (batch_size, x.shape[1]))
        yb = jax.lax.dynamic_slice(y, (start, 0), (batch_size, y.shape[1]))

        loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
        updates, opt_state = optimizer.update(grads, opt_state, params)

        if slow_mask is not None:
            def scale_update(u, m):
                if isinstance(u, jnp.ndarray) and isinstance(m, jnp.ndarray):
                    m = m.astype(u.dtype)  # m in {0.0,1.0}
                    return u * (gamma * m + (1.0 - m))
                return u
            updates = jax.tree_util.tree_map(scale_update, updates, slow_mask)

        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), jnp.arange(batches))
    return params, opt_state, losses


# ========= Model =========

@dataclass
class Model:
    input_dim: int
    output_dim: int
    params: object
    forward: object
    slow_mask: object = None  # persists across train() calls

    @staticmethod
    def init(params, forward, input_dim=None, output_dim=None):
        return Model(
            input_dim=input_dim,
            output_dim=output_dim,
            params=params,
            forward=forward,
            slow_mask=None,
        )

    def assert_data_shape(self, x, y):
        n = x.shape[0]
        if self.input_dim and (x.shape != (n, self.input_dim)):
            raise ValueError(f"Input must be of shape {(n, self.input_dim)}, not {x.shape}")
        if self.output_dim and (y.shape != (n, self.output_dim)):
            raise ValueError(f"Output must be of shape {(n, self.output_dim)}, not {y.shape}")

    # --- Training ---
    def train(
        self,
        train_x,
        train_y,
        epochs=10,
        batch_size=128,
        optimizer=optax.sgd(learning_rate=0.01),
        cost=crossentropy_cost,
        return_score=False,     # Returns a list of losses per batch
        opt_state=None,
        evaluate=None,          # tuple (tx, ty) to evaluate after each epoch
        seed=42,
        batches=None,
        verbose=True,
        l2=False,
        l2_eps=1e-4,
        eval_fn=None,
        # slow-down controls:
        p_slow=0.2,             # fraction of params to slow next epoch
        gamma=0.3,              # scale factor for slowed updates (0<gamma<1)
        weights_only=False,     # slow only weights (2D leaves) if True
    ):
        n = train_x.shape[0]
        if opt_state is None:
            opt_state = optimizer.init(self.params)

        self.assert_data_shape(train_x, train_y)

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

        slow_mask = self.slow_mask  # pick up persisted mask (may be None)

        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")

            # new shuffle each epoch
            key = jax.random.PRNGKey(seed + epoch)
            train_x = jax.random.permutation(key, train_x, axis=0)
            train_y = jax.random.permutation(key, train_y, axis=0)

            prev_params = self.params  # snapshot before updates

            self.params, opt_state, loss = train_epoch(
                params=self.params,
                opt_state=opt_state,
                x=train_x,
                y=train_y,
                optimizer=optimizer,
                loss_fn=loss_fn,
                batches=batches,
                batch_size=batch_size,
                slow_mask=slow_mask,
                gamma=gamma
            )

            if return_score:
                scores.append(jnp.mean(loss))

            if evaluate:
                loss, _ = jax.value_and_grad(eval_fn)(self.params, tx, ty)
                scores.append(loss)
                if verbose:
                    print(f"Eval loss: {loss}")

            # Build mask to slow the biggest changers in the NEXT epoch
            slow_mask = build_slow_mask(prev_params, self.params, p=p_slow, weights_only=weights_only)

        # persist mask across calls
        self.slow_mask = slow_mask

        if return_score:
            return scores, opt_state
        return opt_state

    # --- Inference & metrics ---
    def loss(self, tx, ty, cost=crossentropy_cost):
        loss_fn = _gen_loss_function(self.forward, cost)
        loss, _ = jax.value_and_grad(loss_fn)(self.params, tx, ty)
        return loss

    def accuracy(self, test_x, test_y):
        self.assert_data_shape(test_x, test_y)
        a = self.forward(self.params, test_x)
        a_label = jnp.argmax(a, axis=1)
        t_label = jnp.argmax(test_y, axis=1)
        return jnp.sum(a_label == t_label) / test_x.shape[0] * 100

    def evaluate(self, a):
        return self.forward(self.params, a)

    # --- Persistence ---
    def save(self, path, overwrite=False):
        mode = "wb" if overwrite else "xb"
        with open(path, mode) as f:
            pickle.dump(self, f)

    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
