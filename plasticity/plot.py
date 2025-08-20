import jax.numpy as jnp
import matplotlib.pyplot as plt


if __name__ == '__main__':
    a = jnp.load('sgd.npy')
    b = jnp.load('adamw.npy')

    plt.plot(a, label='sgd lr=0.1')
    plt.plot(b, label='adamw lr=5e-5, w=0.1')
    plt.legend()
    plt.show()