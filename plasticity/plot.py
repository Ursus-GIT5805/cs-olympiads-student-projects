import jax.numpy as jnp
import matplotlib.pyplot as plt


if __name__ == '__main__':
    a = jnp.load('sgd.npy')
    b = jnp.load('adamw.npy')
    plt.plot(a, label='bright')
    plt.plot(b, label='normal')
    plt.legend()
    plt.show()