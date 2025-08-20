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
import math
def getepochsforstudent(epoch,teacher_epochs,total_student_epochs,minepoch):
    return int(minepoch + (total_student_epochs-minepoch) * 0.5 * (1 - math.cos(math.pi * epoch/(teacher_epochs-1))))

if __name__ == '__main__':
    plt.ion()
    train_data, test_data = loader.load_mnist_raw()

    _, ax = plt.subplots()
    (t_loss,) = ax.plot([], label="Teacher loss")
    (s_loss,) = ax.plot([], label="LIVE student loss")

    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")

    ax.legend()

    eras = 20
    student_epochs = 15
    student_final_epochs = eras*student_epochs
    noise_amount_step = 40000
    batch_size = 100

    key = jax.random.PRNGKey(69420)

    model_teacher = presets.Resnet1_mnist(key)
    model_student_along = presets.Resnet1_mnist(key)
    model_student_final = presets.Resnet1_mnist(key)
    optimizer = optax.sgd(learning_rate=0.1)
    opt_state=optimizer.init(model_student_along.params)



    train_teacher_x, train_teacher_y = train_data
    # train_student_x, _ = train_student
    test_x, test_y = test_data
    random_noise = jax.random.uniform(key, shape=(noise_amount_step * eras, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))

    key2 = jax.random.PRNGKey(69)
    random_noise_test = jax.random.uniform(key2, shape=(40000, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))

    student_epochs_along_divergence = []
    accuracies = []

    loss = []
    loss_teacher = []
    # model_student_along.resetsubset()

    for era in range(eras):
        print("Teacher epochs {}/{}".format(era+1, eras))

        model_teacher.train(
            train_teacher_x, train_teacher_y,
            epochs=1, batch_size=batch_size,
            optimizer=optax.sgd(learning_rate=0.1),
            return_score=False,
            key=jax.random.PRNGKey(random.randint(0, int(1e7))),
            # gamma=1,
            # p_slow=0
        )
        l = model_teacher.loss(train_teacher_x, train_teacher_y)
        loss_teacher.append(l)

        t_loss.set_xdata(np.arange(len(loss_teacher)))
        t_loss.set_ydata(loss_teacher)
        # modelstudentlive = presets.Resnet1_mnist(key)
        # for i in range(era+1):
        print("Live student epochs:")
        random_noise_step = random_noise[(era % 30)*noise_amount_step:((era%30)+1)*noise_amount_step]
        print(random_noise_step.device)
        train_student_y = model_teacher.forward(model_teacher.params, random_noise_step)
        opt_state = model_student_along.train(
            random_noise_step, train_student_y,
            epochs=student_epochs, batch_size=batch_size,
            optimizer = optimizer,
            l2=False,
            l2_eps=1e-6,
            opt_state=opt_state,
            key=jax.random.PRNGKey(random.randint(0, int(1e7))
        ))
        l = model_student_along.loss(train_teacher_x, train_teacher_y)
        loss.append(l)

        s_loss.set_xdata(np.arange(len(loss)))
        s_loss.set_ydata(loss)

        teacher_data = model_teacher.forward(model_teacher.params, random_noise_test)

        along_student_acc = model_student_along.accuracy(test_x, test_y)
        accuracies.append(along_student_acc/100)
        # deads = model_student_along.deads(model_student_along.params,random_noise_test)
        # print(deads)
        along_student_data = model_student_along.forward(model_student_along.params, random_noise_test)
        div_stud_along_teacher = kl_divergence(q=along_student_data, p=teacher_data)
        student_epochs_along_divergence.append(div_stud_along_teacher)

        # ax.relim()
        # ax.autoscale_view()

        # plt.draw()
        # plt.pause(0.01)


    # plt.ioff()
    # plt.show()

    # print("After student epochs:")

    # acc_train = model_teacher.accuracy(train_teacher_x, train_teacher_y)
    # acc_test = model_teacher.accuracy(test_x, test_y)
    # print("Accuracy teacher on training data: {}%".format(acc_train))
    # print("Accuracy teacher on test data: {}%".format(acc_test))

    # acc_train = model_student_along.accuracy(train_teacher_x, train_teacher_y)
    # acc_test = model_student_along.accuracy(test_x, test_y)
    # print("Accuracy live student on training data: {}%".format(acc_train))
    # print("Accuracy live student on test data: {}%".format(acc_test))
    # print([float(x) for x in student_epochs_along_divergence])

    # plt.plot(loss, label="along student")
    # plt.plot(loss_teacher, label="teacher")

    # plt.show()
print([float(x) for x in student_epochs_along_divergence])