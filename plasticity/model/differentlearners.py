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

# unused
def reset_weights(params, key, p=0.001):
    """
    Resets a certain percentage of weights
    """
    iterator = zip(params, jax.random.split(key, len(params)))

    out = []
    for (w, b), k in iterator:
        wk1, wk2, bk1, bk2 = jax.random.split(k, 4)

        w1 = jax.random.bernoulli(wk1, p=p, shape=w.shape)
        wr = jax.random.normal(wk2, w.shape)
        wri = 1 - wr
        w2 = (w*wr) + (w1*wri)

        b1 = jax.random.bernoulli(bk1, p=p, shape=b.shape)
        br = jax.random.normal(bk2, b.shape)
        bri = 1 - br
        b2 = (b*br) + (b1*bri)

        param = (w2, b2)
        out.append(param)

    return out

if __name__ == '__main__':
    plt.ion()

    seed = random.randint(0, int(1e9))
    print("Seed:", seed)

    eras = 30
    student_epochs = 15
    noise_amount_step = 40000

    teacher_batch = 100
    batch_size = 100

    # ---

    lr_teacher = 0.1
    lr=0.5

    lr2=0.1
    momentum0 = 0.6
    momentum1 = 0.3
    momentum2 = 0.9
    momentum3 = 0.5
    wd = 0.02
    epoch_lines = []

    # ---

    key = jax.random.PRNGKey(seed)

    model_teacher = presets.Resnet1_mnist(key)

    live_students = []
    optimizers = []
    labels = []

    for lr, mom in [(0.175, 0.875), (0.2, 0.8), (0.2, None)]:
    # for lr in [0.2, 0.5]:
        # for mom in [0.0, 0.8]:
        optimizers.append( optax.sgd(learning_rate=lr, momentum=mom) )
        labels.append(f"sgd (lr={lr}, momentum={mom})")

    for i in range(len(labels)):
        live_students.append( presets.Resnet1_mnist(key) )

    after_student = presets.Resnet1_mnist(key)

    assert len(live_students) == len(optimizers)
    assert len(live_students) == len(labels)

    train_data, test_data = loader.load_mnist_raw()
    train_x, train_y = train_data
    test_x, test_y = test_data

    divergences = []
    accuracies = []
    w = []
    opt_states = []

    # Create plots
    fig_kl, ax_kl = plt.subplots()
    ax_kl.set_xlabel("Epochs")
    ax_kl.set_ylabel("KL Divergence")

    fig_acc, ax_acc = plt.subplots()
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Accuracy over test cases")

    fig_w, ax_w = plt.subplots()
    ax_w.set_xlabel("Epochs")
    ax_w.set_ylabel("Total weight")

    # Line data
    divergence_lines = []
    accuracies_lines = []
    weights_lines = []

    for i in range(len(live_students)):
        divergences.append([])
        accuracies.append([])
        w.append([])

        (kl, ) = ax_kl.plot([], label=labels[i])
        (accl,) = ax_acc.plot([], label=labels[i])
        (wl,) = ax_w.plot([], label=labels[i])

        divergence_lines.append(kl)
        accuracies_lines.append(accl)
        weights_lines.append(wl)

        opt_states.append( optimizers[i].init(live_students[i].params) )

    accs_teacher=[]
    (t_acc,) = ax_acc.plot([], label="Teacher")

    ax_kl.legend()
    ax_acc.legend()
    ax_w.legend()

    info = f"Seed: {seed}"
    ax_kl.text(0, 0, info)
    ax_acc.text(0, 0, info)
    ax_w.text(0, 0, info)

    # Start training
    for era in range(eras):
        print("Era {}/{}".format(era+1, eras))

        print("Teacher learning")
        model_teacher.train(
            train_x, train_y,
            epochs=1, batch_size=teacher_batch,
            optimizer=optax.sgd(learning_rate=lr_teacher),
        )

        accs_teacher.append( model_teacher.accuracy(test_x, test_y) )
        t_acc.set_xdata([student_epochs*x for x in range(len(accs_teacher))])
        t_acc.set_ydata(accs_teacher)

        key2 = jax.random.PRNGKey(seed+era)
        random_noise_test = jax.random.uniform(key2, shape=(noise_amount_step, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))
        teacher_data = model_teacher.forward(model_teacher.params, random_noise_test)

        print("Live student epochs:")
        # Student epochs
        for student_epoch, key in enumerate(jax.random.split(key2, student_epochs)):
            print("Epoch: {}/{}".format(student_epoch+1, student_epochs))

            noise = jax.random.uniform(key, shape=(noise_amount_step, 784), minval=-10.0, maxval=10.0)
            noise_label = model_teacher.evaluate(noise)

            for i, model in enumerate(live_students):
                opt_states[i] = model.train(
                    random_noise_test, teacher_data,
                    epochs=1, batch_size=batch_size,
                    optimizer=optimizers[i],
                    opt_state=opt_states[i],
                    # l2=True,
                    # l2_eps=0.5*(1e-6),
                )

                divergences[i].append( kl_divergence(q=model.evaluate(noise), p=noise_label) )
                divergence_lines[i].set_xdata(np.arange(len(divergences[i])))
                divergence_lines[i].set_ydata(divergences[i])

                accuracies[i].append( model.accuracy(test_x, test_y) )
                accuracies_lines[i].set_xdata(np.arange(len(accuracies[i])))
                accuracies_lines[i].set_ydata(accuracies[i])

                w[i].append( mean_weights(model.params) )
                weights_lines[i].set_xdata(np.arange(len(w[i])))
                weights_lines[i].set_ydata(w[i])

                print("Divergence [{}]: {}".format(labels[i], divergences[i][-1]))

                time.sleep(0.25)

            # =====

            ax_kl.relim()
            ax_kl.autoscale_view()

            ax_acc.relim()
            ax_acc.autoscale_view()

            ax_w.relim()
            ax_w.autoscale_view()

            plt.draw()
            plt.pause(0.01)


    # Print accuracies
    acc_train = model_teacher.accuracy(train_x, train_y)
    acc_test = model_teacher.accuracy(test_x, test_y)
    print("Accuracy teacher on training data: {}%".format(acc_train))
    print("Accuracy teacher on test data: {}%".format(acc_test))

    acc_train = model_student_along.accuracy(train_x, train_y)
    acc_test = model_student_along.accuracy(test_x, test_y)
    print("Accuracy live student on training data: {}%".format(acc_train))
    print("Accuracy live student on test data: {}%".format(acc_test))

    acc_train = model_student_along2.accuracy(train_x, train_y)
    acc_test = model_student_along2.accuracy(test_x, test_y)
    print("Accuracy live student2 on training data: {}%".format(acc_train))
    print("Accuracy live student2 on test data: {}%".format(acc_test))

    plt.ioff()
    plt.show()
