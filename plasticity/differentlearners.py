import math
import jax
import jax.numpy as jnp
import optax
import os

import loader
from model import kl_divergence, crossentropy_cost, gen_loss_function

import presets
from plotter import Plot, Plothandler

import matplotlib.pyplot as plt
import random


def mean_weights(params):
    x = sum(jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params))
    return float(x)


def weight_diff(params):
    x = sum(
        jnp.mean(jnp.abs(jnp.max(p) - jnp.min(p)))
        for p in jax.tree_util.tree_leaves(params)
    )
    return float(x)


if __name__ == "__main__":
    # --- Variables ---
    seed = random.randint(0, int(1e9))

    eras = 20
    student_epochs = 10
    noise_amount_step = 40000

    teacher_batch = 125
    batch_size = 125

    lr_teacher = 4e-4
    wd_teacher = 1e-4

    optimizer_teacher = optax.adamw(learning_rate=lr_teacher, weight_decay=wd_teacher)
    label_teacher = f"teacher (adamw; lr={lr_teacher}, wd={wd_teacher})"

    optimizers = []
    labels = []

    # Create students with sgd
    for lr, mom in [(0.2, 0.8)]:
        optimizers.append(optax.sgd(learning_rate=lr, momentum=mom))
        labels.append(f"sgd (lr={lr}, momentum={mom})")

    # Create students using adamw
    for lr, wd in [(0.001, 0.1), (0.001, 0.0005)]:
        optimizers.append(optax.adamw(learning_rate=lr, weight_decay=wd))
        labels.append(f"adamw (lr={lr}, wd={wd})")

    # ---

    assert len(labels) == len(optimizers), (
        "Number of labels and optimizers must be equals"
    )

    plt.ion()
    print("Seed:", seed)

    key = jax.random.PRNGKey(seed)

    model_teacher = presets.Resnet1_mnist(key)
    opt_state_teacher = optimizer_teacher.init(model_teacher.params)

    # Initialise Studens
    live_students = []
    opt_states = []
    for i in range(len(labels)):
        live_students.append(presets.Resnet1_mnist(key))
        opt_states.append(optimizers[i].init(live_students[i].params))

    loss_fn = gen_loss_function(model_teacher.forward, crossentropy_cost)

    train_data, test_data = loader.load_mnist_raw()
    train_x, train_y = train_data
    test_x, test_y = test_data

    plots = Plothandler()

    title = f"Seed {seed}\n{os.path.basename(__file__)}"

    plots["kl"] = Plot(
        title=title,
        ylabel="KL divergence",
        xlabel="Epochs",
    )
    plots["acc"] = Plot(
        title=title,
        ylabel="Accuracy",
        xlabel="Epochs",
    )
    plots["w"] = Plot(
        title=title,
        ylabel="Sum over absolute weights",
        xlabel="Epochs",
    )

    plt.show(block=False)

    # Start training
    for era, key2 in enumerate(jax.random.split(key, eras)):
        print("Era {}/{}".format(era + 1, eras))

        opt_state_teacher = model_teacher.train(
            train_x,
            train_y,
            epochs=1,
            batch_size=teacher_batch,
            optimizer=optimizer_teacher,
            opt_state=opt_state_teacher,
            verbose=False,
            loss_fn=loss_fn,
            key=key,
        )

        x_pos = student_epochs * era
        plots["acc"].append(
            label_teacher, model_teacher.accuracy(test_x, test_y), x=x_pos
        )
        plots["w"].append(label_teacher, mean_weights(model_teacher.params), x=x_pos)

        teacher_label = model_teacher.evaluate(train_x)

        # Student epochs
        for student_epoch, key in enumerate(jax.random.split(key2, student_epochs)):
            print("Student epoch: {}/{}".format(student_epoch + 1, student_epochs))

            keys = jax.random.split(key, len(live_students))

            for i, (model, key) in enumerate(zip(live_students, keys)):
                name = labels[i]

                k1, k2 = jax.random.split(key, 2)

                noise = jax.random.uniform(
                    k1,
                    shape=(noise_amount_step, 784),
                    minval=-math.sqrt(3),
                    maxval=math.sqrt(3),
                )
                teacher_noise_out = model_teacher.evaluate(noise)

                noise_train = jax.random.uniform(
                    k2,
                    shape=(noise_amount_step, 784),
                    minval=-math.sqrt(3),
                    maxval=math.sqrt(3),
                )
                noise_train_label = model_teacher.evaluate(noise_train)

                opt_states[i] = model.train(
                    noise_train,
                    noise_train_label,
                    epochs=1,
                    batch_size=batch_size,
                    optimizer=optimizers[i],
                    opt_state=opt_states[i],
                    verbose=False,
                    loss_fn=loss_fn,
                    key=key,
                )

                noise_out = model.evaluate(noise)

                plots["kl"].append(
                    name, kl_divergence(q=noise_out, p=teacher_noise_out)
                )
                plots["acc"].append(name, model.accuracy(test_x, test_y))
                plots["w"].append(name, mean_weights(model.params))

            # =====

            plots.draw()

    plt.ioff()
    plt.show()
