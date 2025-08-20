import math
import jax
import optax
import numpy as np

import loader
from linear import *
from model import Model
from model import batch_norm
from model import kl_divergence
from model import squaredmean_cost
import presets
from plotter import *

import matplotlib.pyplot as plt
import copy
import random
import time

# @jax.jit
def mean_weights(params):
    x = sum(jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params))
    return float(x)

# @jax.jit
def weight_diff(params):
    x = sum(jnp.mean(jnp.abs(jnp.max(p) - jnp.min(p))) for p in jax.tree_util.tree_leaves(params))
    return float(x)

def model_diff(model1, model2):
    out = 0

    for (w1, b1), (w2, b2) in zip(model1.params, model2.params):
        diff_w = optax.losses.cosine_similarity(w1.reshape(-1), w2.reshape(-1))
        diff_b = optax.losses.cosine_similarity(b1, b2)
        out = diff_w
    return out

if __name__ == '__main__':
    plt.ion()

    seed = random.randint(0, int(1e9))
    print("Seed:", seed)

    # --- Constants ---
    eras = 20
    student_epochs = 10
    noise_amount_step = 40000

    teacher_batch = 125
    batch_size = 125

    lr_teacher = 0.1
    wd_teacher = 0.0001

    # ---

    key = jax.random.PRNGKey(seed)

    model_teacher = presets.Resnet1_mnist(key)
    # optimizer_teacher = optax.adamw(learning_rate=lr_teacher, weight_decay=wd_teacher)
    optimizer_teacher = optax.sgd(learning_rate=lr_teacher)
    opt_state_teacher = optimizer_teacher.init(model_teacher.params)

    # Create learning methods + labels for the students
    optimizers = []
    labels = []

    for lr, mom in [(0.1, None), (0.2, 0.8)]:
        optimizers.append( optax.sgd(learning_rate=lr, momentum=mom) )
        labels.append(f"sgd (lr={lr}, momentum={mom})")

    for lr, wd in [(0.0001, 0.1), (0.0001, 0.0005)]:
        optimizers.append( optax.adamw(learning_rate=lr, weight_decay=wd) )
        labels.append(f"adamw (lr={lr}, wd={wd})")

    # for lr in [0.2, 0.3, 0.5]:
        # optimizers.append( optax.adam(learning_rate=lr) )
        # labels.append(f"adam (lr={lr})")

    assert len(labels) == len(optimizers)

    # Initialise Studens
    live_students = []
    opt_states = []
    for i in range(len(labels)):
        live_students.append( presets.Resnet1_mnist(key) )
        opt_states.append( optimizers[i].init(live_students[i].params) )


    train_data, test_data = loader.load_mnist_raw()
    train_x, train_y = train_data
    test_x, test_y = test_data

    plots = Plothandler()

    plots["kl"] = Plot(
        ylabel="KL divergence",
        xlabel="Epochs",
    )
    plots["acc"] = Plot(
        ylabel="Accuracy",
        xlabel="Epochs",
    )
    plots["w"] = Plot(
        ylabel="Sum over absolute weights",
        xlabel="Epochs",
    )

    plt.show(block=False)

    # Start training
    for era, key2 in enumerate(jax.random.split(key, eras)):
        print("Era {}/{}".format(era+1, eras))

        opt_state_teacher = model_teacher.train(
            train_x, train_y,
            epochs=1, batch_size=teacher_batch,
            optimizer=optimizer_teacher,
            opt_state=opt_state_teacher,
            verbose=False,
            key=key,
        )

        x_pos = student_epochs*era
        plots["acc"].append("teacher", model_teacher.accuracy(test_x, test_y), x=x_pos)
        plots["w"].append("teacher", mean_weights(model_teacher.params), x=x_pos)

        teacher_label = model_teacher.evaluate(train_x)
        # teacher_test_out = model_teacher.evaluate(test_x)

        # Student epochs
        for student_epoch, key in enumerate(jax.random.split(key2, student_epochs)):
            print("Student epoch: {}/{}".format(student_epoch+1, student_epochs))

            keys = jax.random.split(key, len(live_students))

            for i, (model, key) in enumerate(zip(live_students, keys)):
                name = labels[i]

                k1, k2 = jax.random.split(key, 2)

                noise = jax.random.uniform(k1, shape=(noise_amount_step, 784), minval=-1.0, maxval=1.0)
                teacher_noise_out = model_teacher.evaluate(noise)

                noise_train = jax.random.uniform(k2, shape=(noise_amount_step, 784), minval=-1.0, maxval=1.0)
                noise_train_label = model_teacher.evaluate(noise_train)

                opt_states[i] = model.train(
                    noise_train, noise_train_label,
                    epochs=1, batch_size=batch_size,
                    optimizer=optimizers[i],
                    opt_state=opt_states[i],
                    verbose=False,
                    key=key,
                    # l2=True,
                    # l2_eps=0.5*(1e-6),
                )

                noise_out = model.evaluate(noise)
                # test_out = model.evaluate(test_x)

                # kl_o = f"{name} on test data"
                # kl_r = f"{name}"

                # plots["kl"].append(kl_o, kl_divergence(q=test_out, p=teacher_test_out))
                plots["kl"].append(name, kl_divergence(q=noise_out, p=teacher_noise_out))
                plots["acc"].append(name, model.accuracy(test_x, test_y))
                plots["w"].append(name, mean_weights(model.params))

                time.sleep(0.1) # Slow a bit, my CPU is sweating o.O

            # =====

            plots.draw()
            plt.pause(0.5)

    plt.ioff()
    plt.show()
