# file used to optimize hyperparameters of teacher

import jax
import optax
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing import get_context

import loader
import presets
from model import *
from model import _gen_loss_function

teacher_eras = 30

train_data, test_data = loader.load_mnist_raw()
train_x, train_y = train_data
train_x = jax.device_put(train_x)
train_y = jax.device_put(train_y)

test_x, test_y = test_data
test_x = jax.device_put(test_x)
test_y = jax.device_put(test_y)

def train_model(parameters):
    key, loss_fn, lr, wd, bs = parameters

    model_teacher = presets.Resnet1_mnist(key)

    optimizer_teacher = optax.adamw(learning_rate=lr, weight_decay=wd)
    optimizer_teacher_state = optimizer_teacher.init(model_teacher.params)

    loss_fn = _gen_loss_function(model_teacher.forward, loss_fn)

    for era in range(teacher_eras):
        print("Era: {}/{}; lr: {}; wd: {}; bs: {}".format(era+1, teacher_eras, lr, wd, bs))
        key, teacher_train_key = jax.random.split(key)
        model_teacher.params, optimizer_teacher_state, losses = train_epoch(
            params=model_teacher.params,
            opt_state=optimizer_teacher_state,
            x=train_x,
            y=train_y,
            optimizer=optimizer_teacher,
            loss_fn=loss_fn,
            batches=train_x.shape[0]//bs,
            batch_size=bs,
            key=teacher_train_key
        )

    acc_teacher_test = model_teacher.accuracy(train_x, train_y)
    print("adamw lr: {}; wd: {}; bs: {}; acc: {}".format(lr, wd, bs, acc_teacher_test))
    return (lr, wd, bs, acc_teacher_test)

if __name__ == '__main__':
    seed = random.randint(0, int(1e9))

    key = jax.random.PRNGKey(seed)

    models_seeds, test_noise_seed, key = jax.random.split(key, 3)

    learning_rates = [0.0005]
    weight_decay = [0.0001]
    batch_size = [125]

    to_try = []

    # key, loss_fn, lr, mom, bs
    for lr in learning_rates:
        for wd in weight_decay:
            for bs in batch_size:
                to_try.append([models_seeds, crossentropy_cost, lr, wd, bs])

    

    ctx = get_context('spawn')
    results = []

    for trying in to_try:
        results.append(train_model(trying))

    print(results)
    results_sorted = sorted(results, key=lambda x: x[3], reverse=True)
    for lr, wd, bs, acc in results_sorted:
        print("lr: {}; mom: {}; bs: {}, acc: {}".format(lr, wd, bs, acc))
