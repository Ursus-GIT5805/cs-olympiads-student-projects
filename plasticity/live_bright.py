import math
import jax
import optax

import numpy as np

import loader
from linear import *
from model import *
import presets
from plotter import *

import matplotlib.pyplot as plt
import copy
import random
import math


if __name__ == "__main__":
    plt.ion()

    seed = random.randint(0, int(1e9))
    print("Seed:", seed)

    plots = Plothandler()
    plots["loss"] = Plot(
        xlabel="Epochs",
        ylabel="Loss over test data",
    )
    plots["acc"] = Plot(
        xlabel="Epochs",
        ylabel="Accuracy test data",
    )
    plots["kl"] = Plot(
        xlabel="Epochs",
        ylabel="KL divergence",
    )

    # --- Variables ---

    eras = 20
    student_epochs = 15
    noise_amount_step = 40000

    batch_size = 128

    lr = 0.1

    # ---

    orig_key = jax.random.PRNGKey(seed)

    teacher_optimizer = optax.sgd(learning_rate=lr)
    teacher = presets.Resnet1_mnist(orig_key)

    live_student = presets.Resnet1_mnist(orig_key)
    optimizer = optax.sgd(learning_rate=lr)
    # live_student_opt_state = optimizer.init(live_student.params)

    (train_x, train_y), (test_x, test_y) = loader.load_mnist_raw()

    era_keys = jax.random.split(orig_key, eras)

    for era, key in enumerate(era_keys):
        print("Era {}/{}".format(era+1, eras))

        teacher.train(
            train_x, train_y,
            epochs=1, batch_size=128,
            verbose=False,
            optimizer=teacher_optimizer,
            key=key,
        )

        noise = jax.random.uniform(key, shape=(noise_amount_step, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))
        noise_label = teacher.evaluate(noise)

        print("Training live student")
        live_student_opt_state = live_student.train(
            noise, noise_label,
            epochs=student_epochs, batch_size=batch_size,
            verbose=False,
            optimizer=optimizer,
            # opt_state=live_student_opt_state,
            key=key,
        )

        print("Training bright student")
        bright_student = presets.Resnet1_mnist(orig_key)
        # opt_state = optimizer.init(bright_student.params)
        num_epochs = (era+1)*student_epochs
        for i in range(era+1):
            e_noise = jax.random.uniform(era_keys[i], shape=(noise_amount_step, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))
            e_noise_label = teacher.evaluate(e_noise)

            opt_state = bright_student.train(
                e_noise, e_noise_label,
                epochs=student_epochs, batch_size=batch_size,
                optimizer=optimizer,
                verbose=False,
                # opt_state=opt_state,
                key=era_keys[i],
            )

        live_out = live_student.evaluate(noise)
        bright_out = bright_student.evaluate(noise)

        live_label = "live student"
        bright_label = "bright student"

        plots["loss"].append(live_label, live_student.loss(test_x, test_y))
        plots["loss"].append(bright_label, bright_student.loss(test_x, test_y))

        plots["acc"].append(live_label, live_student.accuracy(test_x, test_y))
        plots["acc"].append(bright_label, bright_student.accuracy(test_x, test_y))

        plots["kl"].append(live_label, kl_divergence(noise_label, live_out))
        plots["kl"].append(bright_label, kl_divergence(noise_label, bright_out))

        plots.draw()
        plt.pause(0.2)

    plt.ioff()
    plt.show()
