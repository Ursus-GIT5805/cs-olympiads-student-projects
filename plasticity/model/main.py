import jax
import jax.numpy as jnp
from dataclasses import dataclass

@jax.jit
def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

@jax.jit
def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1-sig)

@jax.tree_util.register_dataclass
@dataclass
class SquaredMeanCost:
    @staticmethod
    def fn(a, y):
        return 0.5*jnp.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)

@jax.tree_util.register_dataclass
@dataclass
class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        return jnp.sum(jnp.nan_to_num(y * jnp.log(a) + (1-y) * jnp.log1p(-a)))

    @staticmethod
    def delta(_z, a, y):
        return a - y

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
    model,
    data,
    label
):
    err_b = [jnp.zeros(b.shape) for b in model.biases]
    err_w = [jnp.zeros(w.shape) for w in model.weights]

    a = data

    activations = [a]
    zs = []

    sizes = 0
    for b, w in zip(model.biases, model.weights):
        z = jnp.matmul(w, a) + b
        a = sigmoid(z)

        zs.append(z)
        activations.append(a)
        sizes += 1

    # TODO make custom cost function at some point
    delta = model.cost.delta(zs[-1], activations[-1], label)
    # delta = activations[-1] - label

    err_b[-1] = delta
    err_w[-1] = jnp.matmul(delta, activations[-2].transpose())

    for l in range(2, sizes):
        z = zs[-l]
        delta = jnp.dot(model.weights[-l+1].transpose(), delta) * sigmoid_prime(z)
        err_b[-l] = delta
        err_w[-l] = jnp.dot(delta, activations[-l-1].transpose())

    return (err_b, err_w)

@jax.jit
def update_to_batch(
    model,
    batch,
    eta,
):
    err_b = [jnp.zeros(b.shape) for b in model.biases]
    err_w = [jnp.zeros(w.shape) for w in model.weights]
    sz = 0

    for data, label in batch:
        c_err_b, c_err_w = backprop(model, data, label)

        err_b = [b1+b2 for b1, b2 in zip(err_b, c_err_b)]
        err_w = [w1+w2 for w1, w2 in zip(err_w, c_err_w)]

        sz = sz+1

    biases = [
        b - (eta/sz) * db for b, db in zip(model.biases, err_b)
    ]

    weights = [
        w - (eta/sz) * dw for w, dw in zip(model.weights, err_w)
    ]

    return biases, weights

@jax.jit
def feedforward(model, a):
    for b, w in zip(model.biases, model.weights):
        z = jnp.matmul(w, a) + b
        a = sigmoid(z)
    return a

@jax.tree_util.register_dataclass
@dataclass
class Model:
    num_layers: int
    biases: list[jnp.ndarray]
    weights: list[jnp.ndarray]
    cost: object # Cost function
    # activation: object # function

    def layered(layers):
        num_layers = len(layers)
        keys_bias = jax.random.split(jax.random.PRNGKey(2), num_layers-1)

        biases = [
            jax.random.normal(k, shape=(shp,1))
            for k, shp in zip(keys_bias, layers[1:])
        ]

        keys_weights = jax.random.split(jax.random.PRNGKey(4), num_layers-1)
        shapes = zip(layers[1:], layers[:-1])

        weights = [
            jax.random.normal(k, shape=(x,y)) / jnp.sqrt(x)
            for k, (x,y) in zip(keys_weights, shapes)
        ]


        return Model(
            num_layers=num_layers,
            cost=CrossEntropyCost(),
            weights=weights,
            biases=biases
        )

    # Feed forward
    def feed_forward(self, a):
        return feedforward(self, a)

    def train(
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
            print("Epoch {}/{}".format(epoch+1, epochs))

            for batch in make_batches(train_data, batch_size):
                self.biases, self.weights = update_to_batch(
                    self,
                    batch,
                    eta
                )

            # if(return_score):
                # scores.append(self.cross_entropy(train_data))

        # if(return_score):
            # return scores

    def evaluate(
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

    def loss(self, test_data):
        summ = 0.0
        for data, label in test_data:
            a = self.feed_forward(data)
            summ += self.cost.fn(a, label)

        return -summ / len(test_data)

if __name__ == "__main__":
    import loader

    nn = Model.layered([784, 200, 100, 10])

    train_data, test_data = loader.load_mnist()

    nn.train(train_data, epochs=5)

    loss = nn.loss(test_data)

    score = nn.evaluate(test_data)
    print("Score: ", score)
    print("Loss: ", loss)
