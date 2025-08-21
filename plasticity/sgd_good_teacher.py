import math
import jax
import optax
import random
import matplotlib.pyplot as plt

import loader
import presets
from model import *
from model import gen_loss_function

# DO NOT CHANGE TEACHER. I TUNED FOR THE BEST POSSIBLE TEACHER (Alessandro Farcaş)
# adamw lr: 0.0005; wd: 0.0001; bs: 125
# Accuracy of teacher on test data: ~97.98%
# Accuracy of teacher on training data: ~99.73%

if __name__ == '__main__':
    # set up random
    seed = random.randint(0, int(1e9))
    key = jax.random.PRNGKey(seed)
    models_seeds, test_noise_seed, key = jax.random.split(key, 3)

    # load data
    train_data, test_data = loader.load_mnist_raw()
    train_x, train_y = train_data
    test_x, test_y = test_data

    # set up teacher
    teacher_eras = 100
    teacher_lr = 0.0005
    teacher_wd = 0.0001
    teacher_bs = 125
    model_teacher = presets.Resnet1_mnist(models_seeds)
    optimizer_teacher = optax.adamw(learning_rate=teacher_lr, weight_decay=teacher_wd)
    optimizer_teacher_state = optimizer_teacher.init(model_teacher.params)
    loss_fn_teacher = gen_loss_function(model_teacher.forward, crossentropy_cost)

    # set up interactive plot
    plt.ion()
    fig, ax_accuracy = plt.subplots()
    fig, ax_kldiv = plt.subplots()

    line_acc_teach, = ax_accuracy.plot([], label="Teacher (adamw, lr=5e-4; wd=1e-4)")
    line_acc_student_live, = ax_accuracy.plot([], label="Live Student (sgd, lr=0.2; momentum=0.8)")
    ax_accuracy.set_xlabel("Epoch")
    ax_accuracy.set_ylabel("Accuracy")

    line_kl_student_live, = ax_kldiv.plot([], label="Live Student (sgd, lr=0.2; momentum=0.8)")
    ax_kldiv.set_xlabel("Epoch")
    ax_kldiv.set_ylabel("KL divergence")

    ax_accuracy.legend()
    ax_kldiv.legend()

    ax_accuracy.grid()
    ax_kldiv.grid()

    ax_accuracy.set_title("Accuracies on test data")
    ax_kldiv.set_title("KL(Teacher||Student)")

    #set up students
    model_student_live = presets.Resnet1_mnist(models_seeds)
    model_student_final = presets.Resnet1_mnist(models_seeds)

    student_epochs = 15
    bright_student_epochs = teacher_eras * student_epochs
    batch_size_student = 100

    test_noise_amount = 10000
    train_noise_amount = 100000

    test_noise = jax.random.uniform(test_noise_seed, shape=(test_noise_amount, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))

    optimizer_student_live = optax.sgd(learning_rate=0.2, momentum=0.8)
    optimizer_student_live_state = optimizer_student_live.init(model_student_live.params)

    loss_fn_student = gen_loss_function(model_student_live.forward, crossentropy_cost)

    # Data collection for plot

    teacher_acc_epoch = []
    teacher_acc = []
    student_acc_epoch = []
    student_acc = []

    student_kl_div_epoch = []
    student_kl_div = []

    for era in range(teacher_eras):
        print("Era: {}/{}".format(era+1, teacher_eras))
        key, teacher_train_key = jax.random.split(key)
        model_teacher.params, optimizer_teacher_state, losses = train_epoch(
            params=model_teacher.params,
            opt_state=optimizer_teacher_state,
            x=train_x,
            y=train_y,
            optimizer=optimizer_teacher,
            loss_fn=loss_fn_teacher,
            batches=train_x.shape[0]//teacher_bs,
            batch_size=teacher_bs,
            key=teacher_train_key
        )

        acc_teacher_test = model_teacher.accuracy(test_x, test_y)
        teacher_acc.append(acc_teacher_test)
        teacher_acc_epoch.append(era * student_epochs)

        test_noise_y_teach = model_teacher.evaluate(test_noise)



        for epoch in range(student_epochs):
            print("Epoch: {}/{}".format(epoch+1, student_epochs))

            key, noise_key = jax.random.split(key)

            train_noise = jax.random.uniform(noise_key, (train_noise_amount, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))
            train_noise_y = model_teacher.evaluate(train_noise)

            key, student_train_key = jax.random.split(key)
            model_student_live.params, optimizer_student_state, losses = train_epoch(
                params=model_student_live.params,
                x=train_noise,
                y=train_noise_y,
                optimizer=optimizer_student_live,
                opt_state=optimizer_student_live_state,
                loss_fn=loss_fn_student,
                batches=train_noise_amount//batch_size_student,
                batch_size=batch_size_student,
                key=student_train_key
            )
            acc_student_test = model_student_live.accuracy(test_x, test_y)
            student_acc.append(acc_student_test)
            student_acc_epoch.append(era * student_epochs + epoch)

            test_noise_y_student = model_student_live.evaluate(test_noise)
            kl_div = kl_divergence(p=test_noise_y_teach, q=test_noise_y_student)
            student_kl_div_epoch.append(era * student_epochs + epoch)
            student_kl_div.append(kl_div)

            line_acc_teach.set_xdata(teacher_acc_epoch)
            line_acc_teach.set_ydata(teacher_acc)
            line_acc_student_live.set_xdata(student_acc_epoch)
            line_acc_student_live.set_ydata(student_acc)

            line_kl_student_live.set_xdata(student_kl_div_epoch)
            line_kl_student_live.set_ydata(student_kl_div)

            ax_accuracy.relim()
            ax_accuracy.autoscale_view()

            ax_kldiv.relim()
            ax_kldiv.autoscale_view()

            plt.draw()
            plt.pause(0.01)
    
    test_noise_y_student = model_student_live.evaluate(test_noise)
    kl_div = kl_divergence(p=test_noise_y_teach, q=test_noise_y_student)

    print(kl_div)
    
    plt.ioff()
    plt.show()