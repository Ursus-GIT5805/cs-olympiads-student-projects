import jax
# import numpy as jnp
import jax.numpy as jnp
from ml_datasets import mnist
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1-sig)

def make_batches(array, sz):
    n = len(array)

    key = jax.random.PRNGKey(4)
    perm = jax.random.permutation(key, n)

    i = 0
    while i+sz < n:
        yield [array[idx] for idx in perm[i:i+sz]]
        i += sz

    if i < n:
        yield [array[idx] for idx in perm[i:]]



@jax.jit
def backprop(
        biases,
        weights,
        data,
        label
):
    err_b = [jnp.zeros(b.shape) for b in biases]
    err_w = [jnp.zeros(w.shape) for w in weights]

    a = data

    activations = [a]
    zs = []

    for b, w in zip(biases, weights):
        z = jnp.matmul(w, a) + b
        a = sigmoid(z)

        zs.append(z)
        activations.append(a)

        #Start to backpropagate

    delta = (activations[-1] - label) 

    err_b[-1] = delta
    err_w[-1] = jnp.matmul(delta, activations[-2].transpose())

    sizes = len(biases) + 1

    for l in range(2, sizes):
        z = zs[-l]
        delta = jnp.matmul(weights[-l+1].transpose(), delta) * sigmoid_prime(z)
        err_b[-l] = delta
        err_w[-l] = jnp.matmul(delta, activations[-l-1].transpose())

    return (err_b, err_w)

@jax.jit
def update_to_batch(
    biases,
    weights,
    batch,
    eta,
):
    err_b = [jnp.zeros(b.shape) for b in biases]
    err_w = [jnp.zeros(w.shape) for w in weights]

    for data, label in batch:
        c_err_b, c_err_w = backprop(biases, weights, data, label)

        err_b = [b1+b2 for b1, b2 in zip(err_b, c_err_b)]
        err_w = [w1+w2 for w1, w2 in zip(err_w, c_err_w)]

    sz = len(batch)

    biases = [
        b - (eta/sz) * db for b, db in zip(biases, err_b)
    ]

    weights = [
        w - (eta/sz) * dw for w, dw in zip(weights, err_w)
    ]

    return biases, weights

@jax.jit
def feedforward(a, biases, weights):
    for b, w in zip(biases, weights):
        z = jnp.matmul(w, a) + b
        a = sigmoid(z)
    return a

@jax.jit
def singlecross(data, label,biases,weights):
    eps = 1e-12
    out = feedforward(data,biases,weights)         
    summ = jnp.sum(jnp.nan_to_num(label * jnp.log(out + eps) +
                             (1.0 - label) * jnp.log1p(-out)))
    return summ
class Model:
    def __init__(self, sizes):
        self.sizes = sizes
        keys_bias = jax.random.split(jax.random.PRNGKey(2), len(sizes)-1)

        self.biases = [
            jax.random.normal(k, shape=(shp,1))
            for k, shp in zip(keys_bias, sizes[1:])
        ]

        keys_weights = jax.random.split(jax.random.PRNGKey(4), len(sizes)-1)
        shapes = zip(sizes[1:], sizes[:-1])

        self.weights = [
            jax.random.normal(k, shape=(x,y)) / jnp.sqrt(x)
            for k, (x,y) in zip(keys_weights, shapes)
        ]

    # Feed forward
    def feed_forward(self, a):
        return feedforward(a, self.biases, self.weights)

    def gradient_descent(
        self,
        train_data,
        epochs=10,
        batch_size=10,
        eta=0.2,
        return_score=False
    ):
        n = len(train_data)
        scores = []
        for epoch in range(epochs):
           

            for batch in make_batches(train_data, batch_size):
                self.biases, self.weights = update_to_batch(
                    self.biases,
                    self.weights,
                    batch,
                    eta
                )
            print("Epoch {}/{}".format(epoch+1, epochs))
            if(return_score):
                scores.append(self.cross_entropy(train_data))
                
        if(return_score):
            return scores




    def eval_labeled(
        self,
        test_data
    ):
        score = 0
        for data, label in test_data:
            out = self.feed_forward(data)

            idx_label = jnp.argmax(label)
            idx_out = jnp.argmax(out)

            score += int(idx_label == idx_out)

        return score / len(test_data) * 100.0
    
    def cross_entropy(self, test_data):
        summ = 0.0
        for data, label in test_data:
            summ+=singlecross(data,label, self.biases,self.weights)
        return -summ / len(test_data)

# =====

# import matplotlib.pyplot as plt

def load_mnist():
    (train_x, train_y), (test_x, test_y) = mnist()

    train_data = [
        (train_x[i].reshape(-1, 1), train_y[i].reshape(-1, 1))
        for i in range(train_x.shape[0])
    ]

    test_data = [
        (test_x[i].reshape(-1, 1), test_y[i].reshape(-1, 1))
        for i in range(test_x.shape[0])
    ]

    return train_data, test_data

def plot_scores(scores):
    plt.plot(scores)
    plt.show()

if __name__ == "__main__":
    print("Loading testdata")
    train_data, test_data = load_mnist()

    print("Training network")
    nn = Model([28*28, 200, 100, 10])
    scores = nn.gradient_descent(train_data, epochs=30, eta=0.5, return_score=True)
    plot_scores(scores)
    print("Testing accuracy")
    acc = nn.eval_labeled(test_data)
    print("Accuracy {}%".format(acc))
    

