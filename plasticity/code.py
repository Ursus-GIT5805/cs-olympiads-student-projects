import random
import jax.numpy as np
import jax
class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.netlen = len(sizes)
       
        # Create a main random key
        key = jax.random.PRNGKey(45)

        # Split keys for biases and weights
        bias_keys = jax.random.split(key, len(sizes) - 1)
        weight_keys = jax.random.split(key, len(sizes) - 1)

        self.biases = [jax.random.normal(k, shape=(y, 1)) for k, y in zip(bias_keys, sizes[1:])]
        self.weights = [
            jax.random.normal(k, shape=(x, y))
            for k, (y, x) in zip(weight_keys, zip(sizes[:-1], sizes[1:]))
        ]
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def feedforward(self, a):
        for b, w in zip(self.biases,self.weights):
            a = self.sigmoid(np.dot(w,a)+b)
        return a
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    def activation_func(self, output, y):
        return output-y
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases,self.weights):
            z = np.dot(w,activation)+b 
            activation = self.sigmoid(z)
            activations.append(activation)
            zs.append(z)
        delta = self.activation_func(activation,y)*self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())
        for l in range(2,self.netlen):
            delta = np.dot(self.weights[-l+1].transpose(),delta)*self.sigmoid_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def update_batch(self,batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [b+nb for b, nb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [w+nw for w, nw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b-eta/len(batch)*nb for b,nb in zip(self.biases, nabla_b)]
        self.weights = [w-eta/len(batch)*nw for w,nw in zip(self.weights,nabla_w)]

            
    def train(self, eta, training_data, batch_size, epochs):
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0,n,batch_size)]
            for batch in batches:
                self.update_batch(batch,eta)
            print(f"Epoch {j} complete")


# --- assumes your Network class is already defined in the same session ---
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds

# 1) Load MNIST (all in memory)
def load_mnist():
    ds_train = tfds.load("mnist", split="train", as_supervised=True, batch_size=-1)
    ds_test  = tfds.load("mnist", split="test",  as_supervised=True, batch_size=-1)

    train = tfds.as_numpy(ds_train)
    test  = tfds.as_numpy(ds_test)

    X_train, y_train = train
    X_test,  y_test  = test

    # Normalize to [0,1] and reshape to column vectors (784,1)
    X_train = (X_train.astype(np.float32) / 255.0).reshape(-1, 784, 1)
    X_test  = (X_test.astype(np.float32)  / 255.0).reshape(-1, 784, 1)

    # One-hot labels for training (10 classes)
    Y_train_oh = jax.nn.one_hot(jnp.array(y_train), 10).astype(jnp.float32).reshape(-1, 10, 1)

    return jnp.array(X_train), Y_train_oh, jnp.array(X_test), jnp.array(y_test)
# --- Load data ---
X_train, Y_train_oh, X_test, y_test = load_mnist()
print(len(X_train))
# Use only the first 100 samples for quick test
X_train, Y_train_oh = X_train, Y_train_oh
X_test, y_test = X_test, y_test

# Build training list for your train() function
train_list = [(X_train[i], Y_train_oh[i]) for i in range(len(X_train))]

# --- Train ---
net = Network(sizes=[784, 64, 10])
net.train(0.5,train_list ,10, 30,)

# --- Evaluate ---
test_list = [(X_test[i], int(y_test[i])) for i in range(len(X_test))]
acc = net.evaluate(test_list) / len(test_list)
print(f"Test accuracy: {acc:.4f}")

