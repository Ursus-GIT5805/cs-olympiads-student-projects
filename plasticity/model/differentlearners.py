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

def mean_weights(params):
    x = sum(jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params))
    return float(x)

if __name__ == '__main__':
    plt.ion()

    seed = random.randint(0, int(1e9))
    print("Seed:", seed)

    teacher_epochs = 50
    student_epochs = 10
    noise_amount_step = 40000

    teacher_batch = 250
    batch_size = 250

    key = jax.random.PRNGKey(seed)

    model_teacher = presets.Resnet1_mnist(key)
    model_student_along = presets.Resnet1_mnist(key)
    model_student_along2 = presets.Resnet1_mnist(key)

    train_data, test_data = loader.load_mnist_raw()
    train_x, train_y = train_data
    test_x, test_y = test_data

    student_epochs_along_divergence = []
    student_epochs_along_divergence2 = []
    accuracies = []
    accuracies2 = []
    w1 = []
    w2 = []

    lr_teacher = 0.1

    lr=0.5
    lr2=0.0005
    wd = 0.02
    epoch_lines = []

    fig_kl, ax_kl = plt.subplots()
    ax_kl.set_xlabel("Epochs")
    ax_kl.set_ylabel("KL Divergence")

    fig_acc, ax_acc = plt.subplots()
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Accuracy over test cases")
 
    fig_w, ax_w = plt.subplots()
    ax_w.set_xlabel("Epochs")
    ax_w.set_ylabel("Total weight")

    label = f"sgd (lr={lr})"
    label2 = f"adamw (lr={lr2}, wd={wd})"

    (l1,) = ax_kl.plot([], label=label)
    (l2,) = ax_kl.plot([], label=label2)

    (accl1,) = ax_acc.plot([], label=label)
    (accl2,) = ax_acc.plot([], label=label2)

    accs_teacher=[]
    (accl3,) = ax_acc.plot([], label="Teacher")

    (lw1,) = ax_w.plot([], label=label)
    (lw2,) = ax_w.plot([], label=label2)

    ax_kl.legend()
    ax_acc.legend()
    ax_w.legend()

    info = f"Seed: {seed}"
    ax_kl.text(0, 0, info)
    ax_acc.text(0, 0, info)
    ax_w.text(0, 0, info)

    plt.show(block=False)

    optimizer=optax.sgd(learning_rate=lr)
    opt_state=optimizer.init(model_student_along.params)

    optimizer2=optax.adamw(learning_rate=lr2, weight_decay=wd)
    opt_state2=optimizer2.init(model_student_along2.params)

    for epoch in range(teacher_epochs):
        print("Teacher epochs {}/{}".format(epoch+1, teacher_epochs))

        epoch_lines.append(len(student_epochs_along_divergence))
        ax_kl.vlines(epoch_lines, 0, 10, color='grey')
        ax_acc.vlines(epoch_lines, 0, 100, color='grey')
        ax_w.vlines(epoch_lines, 0, 1000, color='grey')

        print("Teacher learning")
        model_teacher.train(
            train_x, train_y,
            epochs=1, batch_size=teacher_batch,
            optimizer=optax.sgd(learning_rate=lr_teacher),
            # seed=random.randint(0, int(1e7)),
        )

        accs_teacher.append( model_teacher.accuracy(test_x, test_y) )
        accl3.set_xdata([student_epochs*x for x in range(len(accs_teacher))])
        accl3.set_ydata(accs_teacher)


        key2 = jax.random.PRNGKey(seed+epoch)
        random_noise_test = jax.random.uniform(key2, shape=(noise_amount_step, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))

        teacher_data = model_teacher.forward(model_teacher.params, random_noise_test)

        print("Live student epochs:")
        for student_epoch, key in enumerate(jax.random.split(key2, student_epochs)):
            print("Epoch: {}/{}".format(student_epoch+1, student_epochs))

            opt_state = model_student_along.train(
                random_noise_test, teacher_data,
                epochs=1, batch_size=batch_size,
                optimizer=optimizer,
                opt_state=opt_state,
            )
            opt_state2 = model_student_along2.train(
                random_noise_test, teacher_data,
                epochs=1, batch_size=batch_size,
                optimizer=optimizer2,
                opt_state=opt_state2,
                # l2=True,
            )

            noise = jax.random.uniform(key, shape=(noise_amount_step, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))
            noise_label = model_teacher.evaluate(noise)

            along_student_acc = model_student_along.accuracy(test_x, test_y)
            along_student_acc2 = model_student_along2.accuracy(test_x, test_y)
            accuracies.append(along_student_acc)
            accuracies2.append(along_student_acc2)

            along_student_data = model_student_along.evaluate(noise)
            along_student_data2 = model_student_along2.evaluate(noise)

            div_stud_along_teacher = kl_divergence(q=along_student_data, p=noise_label)
            div_stud_along_teacher2 = kl_divergence(q=along_student_data2, p=noise_label)

            student_epochs_along_divergence.append(div_stud_along_teacher)
            student_epochs_along_divergence2.append(div_stud_along_teacher2)

            l1.set_xdata(np.arange(len(student_epochs_along_divergence)))
            l1.set_ydata(student_epochs_along_divergence)

            l2.set_xdata(np.arange(len(student_epochs_along_divergence2)))
            l2.set_ydata(student_epochs_along_divergence2)

            accl1.set_xdata(np.arange(len(accuracies)))
            accl1.set_ydata(accuracies)

            accl2.set_xdata(np.arange(len(accuracies2)))
            accl2.set_ydata(accuracies2)

            w1.append( mean_weights(model_student_along.params) )
            w2.append( mean_weights(model_student_along2.params) )

            lw1.set_xdata(np.arange(len(w1)))
            lw1.set_ydata(w1)
            lw2.set_xdata(np.arange(len(w2)))
            lw2.set_ydata(w2)

            ax_kl.relim()
            ax_kl.autoscale_view()

            ax_acc.relim()
            ax_acc.autoscale_view()

            ax_w.relim()
            ax_w.autoscale_view()

            plt.draw()
            plt.pause(0.01)


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
