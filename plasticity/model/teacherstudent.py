import loader
import matplotlib.pyplot as plt
from linear import *
from model import *

import random

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    params_teacher = linears_from_array([784, 100, 100, 100, 10])

    params_student = params_teacher.copy()
    original_params = params_teacher.copy()

    def run(params, a):
        a = feedforward_linear(params[0], a)
        a = jax.nn.sigmoid(a)

        a = feedforward_linear(params[1], a)
        # a = jax.nn.sigmoid(a)
        # a = batch_norm(a)
        # a = jax.nn.relu(a)
        # a = jax.nn.sigmoid(a)
        a = batch_norm(a)
        a = jax.nn.relu(a)

        a = feedforward_linear(params[2], a)
        a = jax.nn.relu(a)
        # a = batch_norm(a)
        # a = jax.nn.relu(a)

        a = feedforward_linear(params[3], a)
        a = jax.nn.softmax( a )
        return a

    train_teacher, train_student, test_data = loader.load_mnist_teacher_student()

    train_teacher_x, train_teacher_y = train_teacher
    train_student_x, _ = train_student
    test_x, test_y = test_data

    teacher = Model.init(
        params_teacher,
        jax.jit(run),
    )
    student = Model.init(
        params_student,
        jax.jit(run)
    )
    student2 = Model.init(
        original_params,
        jax.jit(run)
    )

    student_accs = []
    teacher_accs = []
    acc_student = 0
    teacher_epochs = 10
    student_epochs_per_teacher_epoch = 10

    for teacher_epoch in range(teacher_epochs):
        print(f"Global epoch  {teacher_epoch}:")
        print("Teacher Epochs:")

        teacher.train(
            train_teacher_x, train_teacher_y,
            epochs=1, batch_size=100,
            optimizer=optax.sgd(learning_rate=0.5),
            seed=random.randint(0, int(1e9)),
            batches=100
            # return_score=True,
            # evaluate=(test_x, test_y)
        )

        train_student_y = teacher.evaluate(train_student_x)
        print("Student Epochs:")

        student.train(
            train_student_x, train_student_y,
            epochs=student_epochs_per_teacher_epoch, batch_size=100,
            optimizer=optax.sgd(learning_rate=0.5),
            seed=random.randint(0, int(1e9)),
            batches=10
            # return_score=True,
            # evaluate=(test_x, test_y)
        )
        test_teacher_y = teacher.evaluate(test_x)
        # acc_teacher = teacher.accuracy(test_x,test_y)

        acc_student = student.accuracy(test_x, test_teacher_y)
        print("Accuracy Student: {}%".format(acc_student))
        student_accs.append(acc_student)
        print()

    train_student_y = teacher.evaluate(train_student_x)
    student2.train(
        train_student_x, train_student_y,
        epochs=200, batch_size=100,
        optimizer=optax.sgd(learning_rate=0.5),
        seed=random.randint(0, int(1e9)),
        batches=10
        # return_score=True,
        # evaluate=(test_x, test_y)
    )

    test_teacher_y = teacher.evaluate(test_x)

    acc_student2 = student2.accuracy(test_x,test_teacher_y)
    print("Accuracy Second Student: {}%".format(acc_student2))

    acc_student2_vs_student1 = (acc_student/acc_student2)
    print("Accuracy First Student is {} better than the second Student".format(acc_student2_vs_student1))

    plt.plot(student_accs)
    plt.grid()
    plt.show()
