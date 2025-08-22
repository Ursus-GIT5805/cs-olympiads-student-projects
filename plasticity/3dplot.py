import jax
import optax

import numpy as np

import loader
from model import kl_divergence

import presets

import matplotlib.pyplot as plt
import random


def between(left, right, prec):
    diff = right - left
    for i in range(prec):
        x = left + diff / prec * i
        yield x


if __name__ == "__main__":
    seed = random.randint(0, int(1e9))
    print("Seed:", seed)

    x = []
    y = []
    z = []

    ax = plt.figure().add_subplot(projection="3d")

    ax.set_xlabel("learning rate", fontsize=20)
    ax.set_ylabel("momentum", fontsize=20)
    ax.set_zlabel("KL", fontsize=20)

    # X, Y, Z = axes3d.get_test_data(0.05)
    # print(Z)
    # exit(0)

    # --- Compute data ---

    train_data, test_data = loader.load_mnist_raw()
    train_x, train_y = train_data
    test_x, test_y = test_data

    key = jax.random.PRNGKey(seed)
    model_teacher = presets.Resnet2_mnist(key)
    model_teacher.train(
        train_x,
        train_y,
        epochs=1,
        batch_size=250,
        optimizer=optax.adamw(learning_rate=5e-4),
    )

    noise_size = 60000

    noise = jax.random.uniform(key, shape=(noise_size, 784), minval=-1.0, maxval=1.0)
    noiset = jax.random.uniform(key, shape=(noise_size, 784), minval=-1.0, maxval=1.0)

    noise_label = model_teacher.evaluate(noise)
    noiset_label = model_teacher.evaluate(noiset)

    lr_l = 1e-5
    lr_r = 1e-4

    m_l = 1e-4
    m_r = 1e-5
    prec = 10

    for vx in between(lr_l, lr_r, prec):
        x.append(vx)
    for vy in between(m_l, m_r, prec):
        y.append(vy)

    for momentum in between(m_l, m_r, prec):
        for lr in between(lr_l, lr_r, prec):
            print("Running {},{}".format(lr, momentum))

            _, key = jax.random.split(key, 2)

            model = presets.Resnet2_mnist(key)
            model.train(
                noise,
                noise_label,
                epochs=5,
                batch_size=250,
                optimizer=optax.adamw(learning_rate=5e-4, weight_decay=momentum),
                # optimizer=optax.sgd(learning_rate=lr, momentum=momentum),
            )

            kl = kl_divergence(q=model.evaluate(noiset), p=noiset_label)
            z.append(min(kl, 0.02))

            print(f"[lr={lr}, wd={momentum}]: {kl}")

    # ---

    z = np.array(z)
    z = z.reshape((len(y), len(x)))

    X, Y = np.meshgrid(x, y)  # shape: (prec, prec)
    ax.plot_surface(X, Y, z, cmap=plt.cm.YlGnBu_r)
    plt.show()
