import random 
import jax.numpy as jnp
import jax
import numpy as np
import ml_datasets

@jax.jit
def sigmoid(z):
        return 1.0/(1.0+jnp.exp(-z))

@jax.jit
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

@jax.jit
def dotproduct(w,activation,b):
    z = jnp.dot(w,activation)+b 
    activation = sigmoid(z)
    return (z,activation)

@jax.jit
def seconddotproduct(delta,act):
    return jnp.dot(delta,act.transpose())

@jax.jit
def backpropstep(delta,weights,activations,z):
    delta = jnp.dot(weights.transpose(),delta)*sigmoid_prime(z)
    updw = seconddotproduct(delta,activations)
    return (delta,updw)

def initialize_weights_biases(sizes):
    key = jax.random.PRNGKey(42)

    # Split keys for biases and weights
    bias_keys = jax.random.split(key, len(sizes) - 1)
    weight_keys = jax.random.split(key, len(sizes) - 1)

    biases = [jax.random.normal(k, shape=(y, 1)) for k, y in zip(bias_keys, sizes[1:])]
    weights = [
        jax.random.normal(k, shape=(x, y))/jnp.sqrt(y)
        for k, (y, x) in zip(weight_keys, zip(sizes[:-1], sizes[1:]))
    ]
    return (weights,biases)

def feedforward(a,weights,biases):
    for b, w in zip(biases,weights):
        _ , a = dotproduct(w,a,b)
    return a

def evaluate(test_data,weights,biases):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(jnp.argmax(feedforward(x,weights,biases)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def backprop(x,y,netlen,weights,biases):
    nabla_b = [jnp.zeros(b.shape) for b in biases]
    nabla_w = [jnp.zeros(w.shape) for w in weights]
    activation = x
    activations = [x]
    zs = []
    for b, w in zip(biases,weights):
        z, activation = dotproduct(w,activation,b)
        activations.append(activation)
        zs.append(z)
    
    delta = (activations[-1]-y)*sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = seconddotproduct(delta,activations[-2])
    for l in range(2,netlen):
        delta, nabla_w[-l] = backpropstep(delta,weights[-l+1],activations[-l-1],zs[-l])
        nabla_b[-l] = delta
    return (nabla_b, nabla_w)

def update_batch(batch,eta,sizes,weights, biases):
    nabla_b = [jnp.zeros(b.shape) for b in biases]
    nabla_w = [jnp.zeros(w.shape) for w in weights]
    for x, y in batch:
        delta_nabla_b, delta_nabla_w = backprop(x,y,len(sizes),weights,biases)
        nabla_b = [b+nb for b, nb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [w+nw for w, nw in zip(nabla_w, delta_nabla_w)]
    biases = [b-eta/len(batch)*nb for b,nb in zip(biases, nabla_b)]
    weights = [w-eta/len(batch)*nw for w,nw in zip(weights,nabla_w)]
    return (weights, biases)

def train(training_data, eta, epochs, batch_size, sizes, weights, biases,test_data=None):
    n = len(training_data)
    for epoch in range(epochs):
        random.shuffle(training_data)
        batches = [training_data[k:k+batch_size] for k in range(0,n,batch_size)]
        for batch in batches:
            weights, biases = update_batch(batch,eta, sizes, weights, biases)
        if(test_data):
            score = evaluate(test_data,weights,biases)/len(test_data)
            print(f"Epoch {epoch} complete with score {score}")
        else:
            print(f"Epoch {epoch} complete")
    return (weights,biases)

import numpy as np

def to_one_hot_from_any(y, num_classes=10):
    y = np.array(y)
    if y.ndim == 0:  # already scalar
        y_int = int(y)
    elif y.ndim == 1 and y.size == 1:  # scalar inside vector
        y_int = int(y[0])
    elif y.ndim == 1 and y.size == num_classes:  # already one-hot
        return jnp.array(y.reshape(num_classes, 1), dtype=jnp.float32)
    else:
        raise ValueError(f"Unexpected label shape {y.shape}")
    v = jnp.zeros((num_classes, 1), dtype=jnp.float32)
    return v.at[y_int, 0].set(1.0)
def main():
    (train_images,train_labels), (test_images, test_labels) = ml_datasets.mnist()
    # normalize & reshape
    train_images = train_images.astype(jnp.float32) 
    train_images = train_images.reshape((-1, 784, 1))
    test_images  = test_images.astype(jnp.float32)
    test_images  = test_images.reshape((-1, 784, 1))

    # build datasets in your expected formats
    training_data = [(jnp.array(x), to_one_hot_from_any(y)) for x, y in zip(train_images, train_labels)]
    test_data     = [(jnp.array(x), int(np.argmax(y)) if y.ndim > 0 else int(y))
                 for x, y in zip(test_images, test_labels)]

    sizes = [784, 60,10]
    weights, biases = initialize_weights_biases(sizes)
    weights, biases = train(training_data,0.5,30,10,sizes,weights,biases,test_data)
    # acc = evaluate(test_data,weights,biases) / len(test_data)
    # print(acc)
    
if __name__ == "__main__":
    main()






