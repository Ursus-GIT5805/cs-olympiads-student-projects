import math
import jax
import optax
import os

import loader
from model import kl_divergence, crossentropy_cost, gen_loss_function

import presets
from plotter import Plot, Plothandler

import matplotlib.pyplot as plt
import random

if __name__ == "__main__":
    # --- Variables ---
    seed = random.randint(0, int(1e9))

    eras = 20
    student_epochs = 10
    noise_amount_step = 40000

    teacher_batch = 125
    batch_size = 125

    train_on_noise = True

    lr = 0.1
    # ---

    print("Seed:", seed)

    key = jax.random.PRNGKey(seed)

    teacher = presets.Resnet1_mnist(key)
    optimizer_teacher = optax.sgd(learning_rate=lr)
    opt_state_teacher = optimizer_teacher.init(teacher.params)

    student = presets.Resnet1_mnist(key)
    optimizer_student = optax.sgd(learning_rate=lr)
    opt_state_student = optimizer_student.init(student.params)

    loss_fn = gen_loss_function(teacher.forward, crossentropy_cost)

    (train_x, train_y), (test_x, test_y) = loader.load_mnist_raw()

    # --- Create plots ---

    plt.ion()
    plt.show(block=False)

    title = f"Seed {seed}\n{os.path.basename(__file__)}\n"

    plots = Plothandler()
    plots["acc"] = Plot(title=title, xlabel="Epochs", ylabel="Accuracy over testcases")

    plots["kl"] = Plot(title=title, xlabel="Epochs", ylabel="KL divergence")

    # ---

    for era, k in enumerate(jax.random.split(key, eras)):
        print("Era {}/{}".format(era + 1, eras))

        k1, k2, kn1, kn2 = jax.random.split(k, 4)

        print("Training teacher")
        opt_state_teacher = teacher.train(
            train_x,
            train_y,
            epochs=1,
            batch_size=teacher_batch,
            optimizer=optimizer_teacher,
            opt_state=opt_state_teacher,
            verbose=False,
            loss_fn=loss_fn,
            key=k1,
        )

        plots["acc"].append(
            "teacher", teacher.accuracy(test_x, test_y), student_epochs * era
        )

        student_x = train_x
        if train_on_noise:
            student_x = jax.random.uniform(
                kn1,
                shape=(noise_amount_step, 784),
                minval=-math.sqrt(3),
                maxval=math.sqrt(3),
            )
        student_y = teacher.evaluate(student_x)

        print("Training student...")
        for i in range(student_epochs):
            opt_state_student = student.train(
                student_x,
                student_y,
                epochs=1,
                batch_size=batch_size,
                optimizer=optimizer_student,
                opt_state=opt_state_student,
                verbose=False,
                loss_fn=loss_fn,
                key=k2,
            )

            # --- Measure divergence ---

            ts_student = student.evaluate(test_x)
            ts_teacher = teacher.evaluate(test_x)
            kl_ts = kl_divergence(ts_teacher, ts_student)

            noise = jax.random.uniform(
                kn2,
                shape=(noise_amount_step, 784),
                minval=-math.sqrt(3),
                maxval=math.sqrt(3),
            )
            noise_student = student.evaluate(noise)
            noise_teacher = teacher.evaluate(noise)
            kl_noise = kl_divergence(noise_teacher, noise_student)

            plots["kl"].append("student on test data", kl_ts)
            plots["kl"].append("student on random noise", kl_noise)

            plots["acc"].append("student", student.accuracy(test_x, test_y))

            plots.draw()

    plt.ioff()
    plt.show()
