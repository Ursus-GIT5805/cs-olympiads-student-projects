import math
import jax
import optax
import loader
from linear import *
from model import Model
from model import batch_norm
from model import crossentropy_cost
from model import squaredmean_cost
from model import _gen_loss_function
from model import train_step
import matplotlib.pyplot as plt
import copy
import random


def create_model(params):
    def run(params, a):
        a = feedforward_linear(params[0], a)

        x1 = a.copy()

        a = jax.nn.sigmoid(a)
        a = feedforward_linear(params[1], a)
        a = batch_norm(a)
        a = jax.nn.relu(a)

        a = feedforward_linear(params[2], a)
        a = batch_norm(a)

        a = a + x1
        a = jax.nn.relu(a)

        a = feedforward_linear(params[3], a)
        a = jax.nn.softmax(a)
        return a

    return Model.init(
        params,
        jax.jit(run),
    )

if __name__ == '__main__':
    teacher_epochs = 5
    student_epochs_ratio = 20
    student_final_epochs = 100
    noise_amount_step = 70000

    key = jax.random.PRNGKey(69420)
    params = [
        linear(784, 100, key),
        linear(100, 100, key),
        linear(100, 100, key),
        linear(100, 10, key),
    ]
    model_teacher = create_model(copy.deepcopy(params))
    model_student_along = create_model(copy.deepcopy(params))
    model_student_final = create_model(copy.deepcopy(params))

    train_data, test_data = loader.load_mnist_raw()
    train_x, train_y = train_data
    test_x, test_y = test_data

    print("Loss {}".format(model_teacher.loss(test_x, test_y)))

    random_noise = jax.random.uniform(key, shape=(noise_amount_step * teacher_epochs, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))

    for epoch in range(teacher_epochs):
        print("teacher")
        model_teacher.train(
            train_x, train_y,
            epochs=1, batch_size=50,
            optimizer=optax.sgd(learning_rate=0.5),
            return_score=False,
            evaluate=(test_x, test_y),
            seed=random.randint(0, int(1e7))
        )

        the_key = jax.random.PRNGKey(epoch)
        random_noise_step = random_noise[epoch*noise_amount_step:(epoch+1)*noise_amount_step]
        #print(random_noise_step)
        train_student_y = model_teacher.evaluate(random_noise_step)
        #print(train_student_y)

        model_student_along.train(
            random_noise_step, train_student_y,
            epochs=student_epochs_ratio, batch_size=50,
            optimizer = optax.sgd(learning_rate=0.5),
            return_score=False,
            evaluate=(test_x, test_y),
        )
    train_student_y_final = model_teacher.evaluate(random_noise)
    model_student_final.train(
        random_noise, train_student_y_final,
        epochs=student_final_epochs, batch_size=50,
        optimizer = optax.sgd(learning_rate=0.5),
        return_score=False,
        evaluate=(test_x, test_y),
    )

    acc_train = model_teacher.accuracy(train_x, train_y)
    acc_test = model_teacher.accuracy(test_x, test_y)
    print("Accuracy Teacher Training: {}%".format(acc_train))
    print("Accuracy Teacher Test: {}%".format(acc_test))

    acc_train = model_student_along.accuracy(train_x, train_y)
    acc_test = model_student_along.accuracy(test_x, test_y)
    print("Accuracy student Training: {}%".format(acc_train))
    print("Accuracy student Test: {}%".format(acc_test))

    acc_train = model_student_final.accuracy(train_x, train_y)
    acc_test = model_student_final.accuracy(test_x, test_y)
    print("Accuracy student final Training: {}%".format(acc_train))
    print("Accuracy student final Test: {}%".format(acc_test))

    random_noise_test = jax.random.uniform(key, shape=(600000, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))

    teacher_data = model_teacher.evaluate(random_noise_test)
    acc_stud_follow_teacher = model_student_along.accuracy(test_x, teacher_data)
    acc_stud_final_teacher = model_student_final.accuracy(test_x, teacher_data)

    print("Accuracy of following student-teacher: {}".format(acc_stud_follow_teacher))
    print("Accuracy of final student-teacher: {}".format(acc_stud_final_teacher))
