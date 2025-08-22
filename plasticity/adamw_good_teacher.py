import math
import jax
import jax.numpy as jnp
import optax
import random
import matplotlib.pyplot as plt

import loader
import presets
from model import train_epoch, kl_divergence, gen_loss_function, crossentropy_cost

from plotter import Plothandler, Plot
import os

# DO NOT CHANGE TEACHER. I TUNED FOR THE BEST POSSIBLE TEACHER (Alessandro Farcaş)
# adamw lr: 0.0005; wd: 0.0001; bs: 125
# Accuracy of teacher on test data: ~97.98%
# Accuracy of teacher on training data: ~99.73%

def mean_weights(params):
    x = sum(jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params))
    return float(x)


if __name__ == "__main__":
    # --- Variables ---
    seed = random.randint(0, int(1e9))

    teacher_eras = 100
    teacher_lr = 0.0005
    teacher_wd = 0.0001
    teacher_bs = 125

    student_lr = 5e-5
    student_wd = 1e-6

    student_epochs = 15
    batch_size_student = 100

    test_noise_amount = 10000
    train_noise_amount = 100000
    # ---

    print("Seed:", seed)

    # set up random
    key = jax.random.PRNGKey(seed)
    models_seeds, test_noise_seed, key = jax.random.split(key, 3)

    # load data
    train_data, test_data = loader.load_mnist_raw()
    train_x, train_y = train_data
    test_x, test_y = test_data

    # set up teacher
    model_teacher = presets.Resnet1_mnist(models_seeds)
    optimizer_teacher = optax.adamw(learning_rate=teacher_lr, weight_decay=teacher_wd)
    optimizer_teacher_state = optimizer_teacher.init(model_teacher.params)
    loss_fn_teacher = gen_loss_function(model_teacher.forward, crossentropy_cost)

    # set up interactive plot
    plt.ion()

    plots = Plothandler()

    title = f"Seed {seed}\n{os.path.basename(__file__)}"

    plots["acc"] = Plot(
        title=title,
        xlabel="Student Epoch",
        ylabel="Accuracy",
    )
    plots["kl"] = Plot(
        title=title,
        xlabel="Student Epoch",
        ylabel="Weights Absolute Sum",
    )
    plots["w"] = Plot(
        title=title,
        xlabel="Student Epoch",
        ylabel="KL divergence",
    )

    # Set up students
    model_student_live = presets.Resnet1_mnist(models_seeds)
    model_student_final = presets.Resnet1_mnist(models_seeds)

    bright_student_epochs = teacher_eras * student_epochs

    test_noise = jax.random.uniform(
        test_noise_seed,
        shape=(test_noise_amount, 784),
        minval=-math.sqrt(3),
        maxval=math.sqrt(3),
    )

    optimizer_student_live = optax.adamw(learning_rate=student_lr, weight_decay=student_wd)
    optimizer_student_live_state = optimizer_student_live.init(
        model_student_live.params
    )

    loss_fn_student = gen_loss_function(model_student_live.forward, crossentropy_cost)

    label_teacher = f"Teacher (adamw, lr={teacher_lr}; wd={teacher_wd})"
    student_label = f"Live Student (adamw, lr={student_lr}; wd={student_wd})"

    for era in range(teacher_eras):
        print("Era: {}/{}".format(era + 1, teacher_eras))

        key, teacher_train_key = jax.random.split(key)
        model_teacher.params, optimizer_teacher_state, losses = train_epoch(
            params=model_teacher.params,
            opt_state=optimizer_teacher_state,
            x=train_x,
            y=train_y,
            optimizer=optimizer_teacher,
            loss_fn=loss_fn_teacher,
            batches=train_x.shape[0] // teacher_bs,
            batch_size=teacher_bs,
            key=teacher_train_key,
        )

        acc_teacher_test = model_teacher.accuracy(test_x, test_y)
        test_noise_y_teach = model_teacher.evaluate(test_noise)
        mnist_y_teach = model_teacher.evaluate(test_x)
        wg_teach_d = mean_weights(model_teacher.params)

        plots["acc"].append(label_teacher, acc_teacher_test, x=era*student_epochs)
        plots["w"].append(label_teacher, wg_teach_d, x=era*student_epochs)

        for epoch in range(student_epochs):
            print("Epoch: {}/{}".format(epoch + 1, student_epochs))

            key, noise_key = jax.random.split(key)

            train_noise = jax.random.uniform(
                noise_key,
                (train_noise_amount, 784),
                minval=-math.sqrt(3),
                maxval=math.sqrt(3),
            )
            train_noise_y = model_teacher.evaluate(train_noise)

            key, student_train_key = jax.random.split(key)
            model_student_live.params, optimizer_student_live_state, losses = (
                train_epoch(
                    params=model_student_live.params,
                    x=train_noise,
                    y=train_noise_y,
                    optimizer=optimizer_student_live,
                    opt_state=optimizer_student_live_state,
                    loss_fn=loss_fn_student,
                    batches=train_noise_amount // batch_size_student,
                    batch_size=batch_size_student,
                    key=student_train_key,
                )
            )

            acc_student_test = model_student_live.accuracy(test_x, test_y)

            test_noise_y_student = model_student_live.evaluate(test_noise)
            kl_div = kl_divergence(p=test_noise_y_teach, q=test_noise_y_student)
            test_mnist_y_student = model_student_live.evaluate(test_x)

            kl_div_mnist = kl_divergence(p=mnist_y_teach, q=test_mnist_y_student)
            wg_student_d = mean_weights(model_student_live.params)

            plots["kl"].append(f"{student_label} on random noise", kl_div)
            plots["kl"].append(f"{student_label} on MNIST", kl_div_mnist)
            plots["w"].append(student_label, wg_student_d)
            plots["acc"].append(student_label, acc_student_test)

            plots.draw()

    test_noise_y_student = model_student_live.evaluate(test_noise)
    kl_div = kl_divergence(p=test_noise_y_teach, q=test_noise_y_student)

    print(kl_div)

    plt.ioff()
    plt.show()
