import random
import numpy as np
import jax
import jax.numpy as jnp
import ml_datasets

# -------------------- activations & helpers --------------------

@jax.jit
def sigmoid(z):
    return 1.0 / (1.0 + jnp.exp(-z))


@jax.jit
def dotproduct(w, activation, b):
    z = jnp.dot(w, activation) + b
    activation = sigmoid(z)
    return (z, activation)


def initialize_weights_biases(sizes):
    key = jax.random.PRNGKey(42)
    bias_keys = jax.random.split(key, len(sizes) - 1)
    weight_keys = jax.random.split(key, len(sizes) - 1)
    params = ([],[])
    # biases: (layer_size, 1), for all layers except input
    biases = [jax.random.normal(k, shape=(y, 1)) for k, y in zip(bias_keys, sizes[1:])]

    # weights: (curr, prev) with fan-in scaling by sqrt(prev)
    # zip(sizes[:-1], sizes[1:]) -> (prev, curr); we flip to (curr, prev) for shapes
    weights = [
        jax.random.normal(k, shape=(x, y)) / jnp.sqrt(y)
        for k, (y, x) in zip(weight_keys, zip(sizes[:-1], sizes[1:]))
    ]

    return (weights,biases)

@jax.jit
def feedforward(a, params):
    weights,biases = params
    for w,b in zip(weights,biases):

        _, a = dotproduct(w, a, b)
    return a

def predict(params, x):
    return feedforward(x, params)

def mse_loss(params, x, y):
    """Per-example 0.5 * ||f(x)-y||^2; shapes: x (784,1), y (10,1)."""
    y_hat = predict(params, x)
    return 0.5 * jnp.sum((y_hat - y) ** 2)

# Per-example grads and batch-vectorized grads
per_example_grads = jax.grad(mse_loss)
batched_grads = jax.vmap(per_example_grads, in_axes=(None, 0, 0))

@jax.jit
def update_batch_auto(params, batch_x, batch_y, lr):
    """
    params is (weights, biases) pytree
    batch_x: (B, 784, 1)
    batch_y: (B, 10, 1)
    """
    grads = batched_grads(params, batch_x, batch_y)  # same structure as params
    mean_grads = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), grads)
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, mean_grads)
    return new_params

def evaluate(test_data, params):
    """
    test_data: list of (x, y_idx) where y_idx is int class.
    Returns number correct.
    """
    correct = 0
    for x, y in test_data:
        out = np.array(feedforward(x, params))  # move to host for argmax
        pred = int(out.argmax(axis=0)) if out.ndim == 2 else int(out.argmax())
        if pred == y:
            correct += 1
    return correct

# -------------------- training loop (uses autodiff) --------------------

def update_batch(batch, eta, params):
    """
    Wraps the jitted autodiff update: stack list-of-tuples into batched arrays.
    """
    batch_x = jnp.stack([x for x, _ in batch], axis=0)  # (B, 784, 1)
    batch_y = jnp.stack([y for _, y in batch], axis=0)  # (B, 10, 1)
    params = update_batch_auto(params, batch_x, batch_y, eta)
    return params

def train(training_data, eta, epochs, batch_size, params, test_data=None):
    n = len(training_data)
    for epoch in range(epochs):
        random.shuffle(training_data)
        batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
        for batch in batches:
            params = update_batch(batch, eta, params)
            print(f"Epoch {epoch} complete")
    if test_data:
        score = evaluate(test_data, params) / len(test_data)
        print(f"score {score}")
    return params

# -------------------- utils & main --------------------

def to_one_hot_from_any(y, num_classes=10):
    y = np.array(y)
    if y.ndim == 0:
        y_int = int(y)
    elif y.ndim == 1 and y.size == 1:
        y_int = int(y[0])
    elif y.ndim == 1 and y.size == num_classes:
        return jnp.array(y.reshape(num_classes, 1), dtype=jnp.float32)
    else:
        raise ValueError(f"Unexpected label shape {y.shape}")
    v = jnp.zeros((num_classes, 1), dtype=jnp.float32)
    return v.at[y_int, 0].set(1.0)

def main():
    (train_images, train_labels), (test_images, test_labels) = ml_datasets.mnist()

    # to float32 and column-vectors
    train_images = train_images.astype(jnp.float32).reshape((-1, 784, 1))
    test_images  = test_images.astype(jnp.float32).reshape((-1, 784, 1))
    # Optional: uncomment to normalize to [0,1]
    # train_images = train_images / 255.0
    # test_images  = test_images / 255.0

    training_data = [(jnp.array(x), to_one_hot_from_any(y)) for x, y in zip(train_images, train_labels)]
    test_data     = [(jnp.array(x), int(y) if np.ndim(y) == 0 else int(np.argmax(y)))
                     for x, y in zip(test_images, test_labels)]

    sizes = [784, 100, 40,  10]
    params = initialize_weights_biases(sizes)

    # Train with autodiff + jit
    params = train(
        training_data=training_data,
        eta=0.5,
        epochs=30,
        batch_size=10,
        params=params,
        test_data=test_data
    )

if __name__ == "__main__":
    main()
