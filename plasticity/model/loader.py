import jax
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

    Returns None on error.
    """

    with open(path, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

        # Convert labels to onehot
        label = data[b'labels']
        label = jnp.eye(10)[jnp.array(label)]

        # Convert batch to values from 0-1
        batch = jnp.array(data[b'data']) / 255

        return batch, label
    return None

def load_mnist_teacher_student():
    (train_x, train_y), (test_x, test_y) = ml_datasets.mnist()
    train_teacher_x = train_x[:27000]
    train_teacher_y = train_y[:27000]
    train_student_x = train_x[27000:]
    train_student_y = train_y[27000:]

    return (train_teacher_x,train_teacher_y), (train_student_x, train_student_y),(test_x,test_y)
