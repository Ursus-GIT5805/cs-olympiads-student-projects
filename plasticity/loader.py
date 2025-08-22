"""
Utility methods to load different datasets.
"""

import jax.numpy as jnp
import pickle
import ml_datasets

def load_mnist():
    (train_x, train_y), (test_x, test_y) = ml_datasets.mnist()

    train_data = [
        (train_x[i].reshape(-1, 1), train_y[i].reshape(-1, 1))
        for i in range(train_x.shape[0])
    ]

    test_data = [
        (test_x[i].reshape(-1, 1), test_y[i].reshape(-1, 1))
        for i in range(test_x.shape[0])
    ]

    return train_data, test_data


def load_mnist_raw():
    return ml_datasets.mnist()


def load_cifar10(path: str):
    """
    Loads the cifar10 batch at PATH.
    Download the cifar10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html.

    Returns None on error.
    """

    with open(path, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")

        # Convert labels to onehot
        label = data[b"labels"]
        label = jnp.eye(10)[jnp.array(label)]

        # Convert batch to values from 0-1
        batch = jnp.array(data[b"data"]) / 255
        batch = batch.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        return batch, label
    return None
