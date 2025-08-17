# import jax
# import jax.numpy as jnp
# import optax
# import random
# import dill as pickle
# import math
# from dataclasses import dataclass
# from functools import partial

# @jax.jit
# def batch_norm(x):
#     mean = jnp.mean(x)
#     var = jnp.var(x)
#     return jnp.nan_to_num((x-mean) / jnp.sqrt(var))

# # =====

# def _flatten_leaves(params):
#     leaves, treedef = jax.tree_util.tree_flatten(params)
#     return leaves, treedef

# def _concat_abs_and_meta(leaves, kind):
#     """
#     kind: 'weight' -> 2D arrays; 'bias' -> 1D arrays
#     Returns:
#       abs_all: concatenated abs values (1D)
#       metas: list of (idx, shape, is_target) to map back
#     """
#     abs_chunks = []
#     metas = []
#     for i, x in enumerate(leaves):
#         is_weight = (x.ndim == 2)
#         is_bias   = (x.ndim == 1)
#         is_target = (is_weight if kind == 'weight' else is_bias)
#         metas.append((i, x.shape, is_target))
#         if is_target:
#             abs_chunks.append(jnp.abs(jnp.ravel(x)))
#     if abs_chunks:
#         abs_all = jnp.concatenate(abs_chunks, axis=0)
#     else:
#         abs_all = jnp.array([], dtype=leaves[0].dtype if leaves else jnp.float32)
#     return abs_all, metas

# def _threshold_for_top_p(abs_all, p):
#     total = abs_all.size
#     k = int(math.floor(p * total))
#     if k <= 0 or total == 0:
#         return None  # no-op
#     # threshold = kth largest value
#     # sort is simplest; for very large arrays you can use jnp.partition
#     sorted_vals = jnp.sort(abs_all)
#     thresh = sorted_vals[-k]  # may include ties
#     return thresh

# def reset_top_by_magnitude(params, key, p=0.2):
#     """
#     Reset the top p-fraction (by absolute value) of:
#       - all WEIGHT elements (2D leaves), using N(0, 1/sqrt(in_features))
#       - all BIAS elements (1D leaves), using N(0, 1)
#     Returns new params.
#     """
#     leaves, treedef = _flatten_leaves(params)

#     # Compute thresholds separately
#     abs_w_all, metas_w = _concat_abs_and_meta(leaves, 'weight')
#     abs_b_all, metas_b = _concat_abs_and_meta(leaves, 'bias')
#     thresh_w = _threshold_for_top_p(abs_w_all, p)
#     thresh_b = _threshold_for_top_p(abs_b_all, p)

#     # PRNG per leaf
#     keys = jax.random.split(key, len(leaves))

#     new_leaves = []
#     for (i, x) in enumerate(leaves):
#         k_leaf = keys[i]
#         if x.ndim == 2 and thresh_w is not None:
#             # weights
#             in_features = x.shape[0]
#             scale = 1.0 / jnp.sqrt(in_features)
#             # mask top-|x| elements
#             mask = jnp.abs(x) >= thresh_w
#             # reinit values for masked positions
#             noise = jax.random.normal(k_leaf, x.shape) * scale
#             x = jnp.where(mask, noise, x)
#             new_leaves.append(x)
#         elif x.ndim == 1 and thresh_b is not None:
#             # biases
#             mask = jnp.abs(x) >= thresh_b
#             noise = jax.random.normal(k_leaf, x.shape)
#             x = jnp.where(mask, noise, x)
#             new_leaves.append(x)
#         else:
#             new_leaves.append(x)

#     return jax.tree_util.tree_unflatten(treedef, new_leaves)


# @jax.jit
# def kl_divergence(p, q):
#     eps=1e-12
#     p = jnp.clip(p, eps, 1.0)
#     q = jnp.clip(q, eps, 1.0)
#     return jnp.mean(p * (jnp.log(p) - jnp.log(q)))

# @jax.jit
# def kl_divergence_cost(a, y):
#     return kl_divergence(y, a)

# # =====

# @jax.jit
# def crossentropy_cost(a, y):
#     eps = 0.001
#     return jnp.mean(-y * jnp.log(a+eps) - (1-y) * jnp.log1p(-a+eps))

# @jax.jit
# def squaredmean_cost(a, y):
#     return jnp.mean( (a-y) ** 2 )

# # ===== Training =====

# def _gen_loss_function(
#         run,
#         cost,
#         l2=False,
#         l2_eps=1e-4
# ):
#     if l2:
#         def loss_fn(params, x, y):
#             a = run(params, x)
#             l2_loss = sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))
#             return cost(a, y) + l2_loss*l2_eps

#     else:
#         def loss_fn(params, x, y):
#             a = run(params, x)
#             return cost(a, y)

#     return jax.jit(loss_fn)

# @partial(jax.jit, static_argnames=('optimizer', 'loss_fn', 'batches', 'batch_size'))
# def train_epoch(params, opt_state, x, y, optimizer, loss_fn, batches, batch_size):

#     def step(carry, batch_idx):
#         params, opt_state = carry
#         start = batch_idx * batch_size
#         xb = jax.lax.dynamic_slice(x, (start, 0), (batch_size, x.shape[1]))
#         yb = jax.lax.dynamic_slice(y, (start, 0), (batch_size, y.shape[1]))

#         loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
#         updates, opt_state = optimizer.update(grads, opt_state, params)
#         params = optax.apply_updates(params, updates)
#         return (params, opt_state), loss

#     (params, opt_state), losses = jax.lax.scan(step, (params, opt_state), jnp.arange(batches))
#     return params, opt_state, losses



# # ===== Model =====

# @dataclass
# class Model:
#     input_dim: int
#     output_dim: int
#     params: object
#     forward: object

#     @staticmethod
#     def init(
#         params,
#         forward,
#         input_dim=None,
#         output_dim=None,
#     ):
#         return Model(
#             input_dim=input_dim,
#             output_dim=output_dim,
#             params=params,
#             forward=forward,
#         )

#     def assert_data_shape(self, x, y):
#         n = x.shape[0]

#         if self.input_dim and (x.shape != (n, self.input_dim)):
#             raise ValueError(
#                 "Input most be of shape {}, not {}"
#                 .format((n, self.input_dim), x.shape)
#             )

#         if self.output_dim and (y.shape != (n, self.output_dim)):
#             raise ValueError(
#                 "Output most be of shape {}, not {}"
#                 .format(n, self.output_dim, y.shape)
#             )
#     # def resetsubset(self):
#     #     newparams=[]
#     #     for param in self.params:
#     #         newparamweights=0
#     #         for w1 in param[0]:
#     #             for w2 in w1:
#     #                 if(random.random()<0.2):
    
#     def model_reset_top(self, p=0.2, seed=0):
#         """
#         Reset top-|value| p fraction of weights and biases (separately).
#         """
#         key = jax.random.PRNGKey(seed)
#         self.params = reset_top_by_magnitude(self.params, key, p=p)

#     def train(
#         self,
#         train_x,
#         train_y,
#         epochs=10,
#         batch_size=128,
#         optimizer=optax.sgd(learning_rate=0.01),
#         cost=crossentropy_cost,
#         return_score=False, # Returns a list of losses per batch
#         opt_state=None,
#         evaluate=None, # Prints a list of losses corresponding to the given test data
#         seed=42,
#         batches=None,
#         verbose=True,
#         l2=False,
#         l2_eps=1e-4,
#         eval_fn=None,
#     ):
#         n = train_x.shape[0]

#         if opt_state == None:
#             opt_state = optimizer.init(self.params)

#         self.assert_data_shape(train_x, train_y)

#         r = n
#         if not batches:
#             batches = train_x.shape[0] // batch_size
#         if not batch_size:
#             batch_size = train_x.shape[0] // batches

#         scores = []
#         loss_fn = _gen_loss_function(self.forward, cost, l2=l2, l2_eps=l2_eps)
#         if not eval_fn:
#             eval_fn = cost
#         eval_fn = _gen_loss_function(self.forward, eval_fn, l2=l2, l2_eps=l2_eps)

#         if evaluate:
#             tx, ty = evaluate
#             self.assert_data_shape(tx, ty)

#         for epoch in range(epochs):
#             if verbose:
#                 print("Epoch {}/{}".format(epoch+1, epochs))

#             key = jax.random.PRNGKey(seed)
#             train_x = jax.random.permutation(key, train_x, axis=0)
#             train_y = jax.random.permutation(key, train_y, axis=0)

#             self.params, opt_state, loss = train_epoch(
#                 params=self.params,
#                 opt_state=opt_state,
#                 x=train_x,
#                 y=train_y,
#                 optimizer=optimizer,
#                 loss_fn=loss_fn,
#                 batches=batches,
#                 batch_size=batch_size
#             )

#             if return_score:
#                 scores.append(jnp.mean(loss))

#             if evaluate:
#                 loss, _ = jax.value_and_grad(eval_fn)(self.params, tx, ty)
#                 scores.append(loss)
#                 print("Loss: {}".format(loss))

#         if return_score:
#             return scores, opt_state

#         return opt_state

#     def loss(
#         self,
#         tx,
#         ty,
#         cost=crossentropy_cost,
#     ):
#         loss_fn = _gen_loss_function(self.forward, cost)
#         loss, _ = jax.value_and_grad(loss_fn)(self.params, tx, ty)
#         return loss

#     def accuracy(
#         self,
#         test_x,
#         test_y,
#     ):
#         self.assert_data_shape(test_x, test_y)

#         a = self.forward(self.params, test_x)

#         a_label = jnp.argmax(a, axis=1)
#         t_label = jnp.argmax(test_y, axis=1)

#         return jnp.sum(a_label == t_label) / test_x.shape[0] * 100

#     def evaluate(self, a):
#         return self.forward(self.params, a)

#     def save(self, path, overwrite=False):
#         mode = "wb" if overwrite else "xb"

#         with open(path, mode) as f:
#             pickle.dump(self, f)

#     def load(path):
#         with open(path, "rb") as f:
#             return pickle.load(f)
# model.py
import math
from dataclasses import dataclass
from functools import partial

import dill as pickle
import jax
import jax.numpy as jnp
import optax


# ========= Helpers =========

def _is_array(x):
    try:
        from jax import Array as _JaxArray
        return isinstance(x, (jnp.ndarray, _JaxArray))
    except Exception:
        return isinstance(x, jnp.ndarray)

def _zeros_like_tree(tree):
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) if _is_array(x) else x, tree)

def _copy_tree(tree):
    def _cp(x):
        if _is_array(x):
            return jnp.array(x, copy=True)
        return x
    return jax.tree_util.tree_map(_cp, tree)


# ========= Normalization & Losses =========

@jax.jit
def batch_norm(x):
    mean = jnp.mean(x)
    var = jnp.var(x)
    return jnp.nan_to_num((x - mean) / jnp.sqrt(var + 1e-8))

@jax.jit
def kl_divergence(p, q):
    eps = 1e-12
    p = jnp.clip(p, eps, 1.0)
    q = jnp.clip(q, eps, 1.0)
    return jnp.mean(jnp.sum(p * (jnp.log(p) - jnp.log(q)), axis=1))

@jax.jit
def kl_divergence_cost(a, y):
    return kl_divergence(y, a)

@jax.jit
def crossentropy_cost(a, y):
    eps = 1e-3
    return jnp.mean(-y * jnp.log(a + eps) - (1 - y) * jnp.log1p(-(a - eps)))

@jax.jit
def softmax_crossentropy_from_probs(a, y, eps=1e-9):
    return -jnp.mean(jnp.sum(y * jnp.log(a + eps), axis=1))

@jax.jit
def softmax_crossentropy_from_logits(logits, y):
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(y * logp, axis=1))

@jax.jit
def distill_kl_from_logits(z_teacher, z_student, T=2.0):
    pt = jax.nn.softmax(z_teacher / T)
    logqs = jax.nn.log_softmax(z_student / T)
    kl = jnp.sum(pt * (jnp.log(jnp.clip(pt, 1e-8, 1.0)) - logqs), axis=1)
    return (T * T) * jnp.mean(kl)

@jax.jit
def squaredmean_cost(a, y):
    return jnp.mean((a - y) ** 2)


# ========= Generic loss wrapper =========

def _gen_loss_function(run, cost, l2=False, l2_eps=1e-4):
    if l2:
        def loss_fn(params, x, y):
            a = run(params, x)
            l2_loss = sum(jnp.sum(p ** 2) for p in jax.tree_util.tree_leaves(params) if _is_array(p))
            return cost(a, y) + l2_loss * l2_eps
    else:
        def loss_fn(params, x, y):
            a = run(params, x)
            return cost(a, y)
    return jax.jit(loss_fn)


# ========= Synaptic Intelligence =========

def _si_penalty(params, theta_star, omega):
    def term(p, ts, om):
        if _is_array(p):
            d = (p - ts)
            return jnp.sum(om * d * d)
        return 0.0
    terms = jax.tree_util.tree_map(term, params, theta_star, omega)
    leaves, _ = jax.tree_util.tree_flatten(terms)
    return jnp.sum(jnp.array([l for l in leaves]))

def _gen_loss_function_with_si(model, run, base_cost, l2=False, l2_eps=1e-4):
    base = _gen_loss_function(run, base_cost, l2=l2, l2_eps=l2_eps)
    if not model.si_enabled:
        return base

    def loss_fn(params, x, y):
        base_loss = base(params, x, y)
        si_loss = _si_penalty(params, model.si_theta_star, model.si_omega)
        return base_loss + model.si_lambda * si_loss

    return jax.jit(loss_fn)


@partial(jax.jit, static_argnames=('optimizer', 'loss_fn', 'batches', 'batch_size'))
def train_epoch(params, opt_state, x, y, optimizer, loss_fn, batches, batch_size,
                si_on=False, si_path=None):

    def step(carry, batch_idx):
        params, opt_state, si_path = carry
        start = batch_idx * batch_size
        xb = jax.lax.dynamic_slice(x, (start, 0), (batch_size, x.shape[1]))
        yb = jax.lax.dynamic_slice(y, (start, 0), (batch_size, y.shape[1]))

        loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
        updates, opt_state = optimizer.update(grads, opt_state, params)

        # SI path integral: ω += g * Δθ
        if si_on:
            def acc_fn(path, g, du):
                if _is_array(du) and _is_array(g):
                    return path + g * du
                return path
            si_path = jax.tree_util.tree_map(acc_fn, si_path, grads, updates)

        params = optax.apply_updates(params, updates)
        return (params, opt_state, si_path), loss

    (params, opt_state, si_path), losses = jax.lax.scan(
        step, (params, opt_state, si_path), jnp.arange(batches)
    )
    return params, opt_state, losses, si_path


# ========= Model =========

@dataclass
class Model:
    input_dim: int
    output_dim: int
    params: object
    forward: object

    # --- SI state ---
    si_enabled: bool = False
    si_lambda: float = 1.0
    si_xi: float = 0.1
    si_theta_star: object = None
    si_omega: object = None
    si_path_contrib: object = None
    si_theta_start: object = None

    @staticmethod
    def init(params, forward, input_dim=None, output_dim=None):
        return Model(
            input_dim=input_dim,
            output_dim=output_dim,
            params=params,
            forward=forward,
        )

    # ===== Synaptic Intelligence controls =====
    def si_enable(self, lam=1.0, xi=0.1):
        self.si_enabled = True
        self.si_lambda = lam
        self.si_xi = xi
        if self.si_theta_star is None:
            self.si_theta_star = _copy_tree(self.params)
        if self.si_omega is None:
            self.si_omega = _zeros_like_tree(self.params)

    def si_start_phase(self):
        if not self.si_enabled:
            return
        self.si_theta_start = _copy_tree(self.params)
        self.si_path_contrib = _zeros_like_tree(self.params)

    def si_end_phase(self):
        if not self.si_enabled:
            return
        assert self.si_theta_start is not None and self.si_path_contrib is not None

        def update_omega(omega, path, theta_end, theta_start):
            delta = theta_end - theta_start
            denom = delta * delta + self.si_xi
            return omega + path / denom

        self.si_omega = jax.tree_util.tree_map(
            update_omega, self.si_omega, self.si_path_contrib, self.params, self.si_theta_start
        )
        self.si_theta_star = _copy_tree(self.params)
        self.si_theta_start = None
        self.si_path_contrib = None

    # ===== Shape check =====
    def assert_data_shape(self, x, y):
        n = x.shape[0]
        if self.input_dim and (x.shape != (n, self.input_dim)):
            raise ValueError(f"Input must be of shape {(n, self.input_dim)}, not {x.shape}")
        if self.output_dim and (y.shape != (n, self.output_dim)):
            raise ValueError(f"Output must be of shape {(n, self.output_dim)}, not {y.shape}")

    # ===== Training =====
    def train(
        self,
        train_x,
        train_y,
        epochs=10,
        batch_size=128,
        optimizer=optax.sgd(learning_rate=0.01),
        cost=crossentropy_cost,
        return_score=False,
        opt_state=None,
        evaluate=None,
        seed=42,
        batches=None,
        verbose=True,
        l2=False,
        l2_eps=1e-4,
        eval_fn=None,
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
        loss_fn = _gen_loss_function_with_si(self, self.forward, cost, l2=l2, l2_eps=l2_eps)
        if not eval_fn:
            eval_fn = cost
        eval_fn_jit = _gen_loss_function_with_si(self, self.forward, eval_fn, l2=l2, l2_eps=l2_eps)

        if evaluate:
            tx, ty = evaluate
            self.assert_data_shape(tx, ty)

        if self.si_enabled and (self.si_path_contrib is None or self.si_theta_start is None):
            self.si_start_phase()

        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")

            key = jax.random.PRNGKey(seed + epoch)
            perm = jax.random.permutation(key, n)
            train_x = train_x[perm]
            train_y = train_y[perm]

            self.params, opt_state, losses, new_si_path = train_epoch(
                params=self.params,
                opt_state=opt_state,
                x=train_x,
                y=train_y,
                optimizer=optimizer,
                loss_fn=loss_fn,
                batches=batches,
                batch_size=batch_size,
                si_on=self.si_enabled,
                si_path=self.si_path_contrib if self.si_enabled else None,
            )

            if self.si_enabled:
                self.si_path_contrib = new_si_path

            if return_score:
                scores.append(jnp.mean(losses))

            if evaluate:
                eval_loss, _ = jax.value_and_grad(eval_fn_jit)(self.params, tx, ty)
                scores.append(eval_loss)
                if verbose:
                    print(f"Eval loss: {eval_loss}")

        if return_score:
            return scores, opt_state
        return opt_state

    # ===== Inference & metrics =====
    def loss(self, tx, ty, cost=crossentropy_cost):
        loss_fn = _gen_loss_function_with_si(self, self.forward, cost)
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

    # ===== Persistence =====
    def save(self, path, overwrite=False):
        mode = "wb" if overwrite else "xb"
        with open(path, mode) as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
