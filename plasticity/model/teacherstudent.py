import loader
import matplotlib.pyplot as plt
from linear import *
from model import *
import math
import random

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    params_teacher = [
        linear(784, 100, key),
        linear(100, 100, key),
        linear(100, 100, key),
        linear(100, 10, key),
    ]


    params_student = params_teacher.copy()
    original_params = params_teacher.copy()

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
        a = jax.nn.softmax( a )
        return a

    train_teacher, train_student, test_data = loader.load_mnist_teacher_student()

    train_teacher_x, train_teacher_y = train_teacher
    train_student_x, _ = train_student
    test_x, test_y = test_data

    forward = jax.jit(run)

    teacher = Model.init(
        params_teacher,
        forward,
    )
    student = Model.init(
        params_student,
        forward,
    )
    student2 = Model.init(
        original_params,
        forward,
    )

    student_accs = []
    latestudents_accs = []
    acc_student = 0
    teacher_epochs = 20
    student_epochs_per_teacher_epoch = 10

    optimizer = optax.sgd(learning_rate=0.5)

    for epoch in range(teacher_epochs):
        print(f"Global epoch  {epoch}:")

        # ===== Teacher training =====
        print("Teacher Epochs:")
        teacher.train(
            train_teacher_x, train_teacher_y,
            epochs=1,
            batch_size=100,
            optimizer=optimizer,
            seed=random.randint(0, int(1e9)),
            batches=10
            # return_score=True,
            # evaluate=(test_x, test_y)
        )
        train_student_y = teacher.evaluate(train_student_x)

        # ===== Student1 training =====
        print("Student Epochs:")
        student.train(
            train_student_x, train_student_y,
            epochs=student_epochs_per_teacher_epoch,
            batch_size=100,
            optimizer=optimizer,
            seed=random.randint(0, int(1e9)),
            batches=10
            # return_score=True,
            # evaluate=(test_x, test_y)
        )

        random_noise_test = jax.random.uniform(key, shape=(6000, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))

        # Measure divergence between teacher and student
        test_teacher_y = teacher.evaluate(random_noise_test)
        test_student_y = student.evaluate(random_noise_test)
        kl_student = kl_divergence(test_teacher_y,test_student_y)

        student_accs.append(kl_student)

        # ===== Student2 training =====
        thparams = original_params.copy()
        studenth = Model.init(
            thparams,
            forward,
        )
        studenth.train(
            train_student_x, train_student_y,
            epochs=student_epochs_per_teacher_epoch*(epoch+1),
            batch_size=100,
            optimizer=optimizer,
            seed=random.randint(0, int(1e9)),
            batches=10
            # return_score=True,
            # evaluate=(test_x, test_y)
        )
        test_student_y = studenth.evaluate(random_noise_test)
        kl_student = kl_divergence(test_teacher_y,test_student_y)

        print("Bright student KL {}".format(kl_student))
        latestudents_accs.append(kl_student)

        print()


    num_epochs = student_epochs_per_teacher_epoch*teacher_epochs
    train_student_y = teacher.evaluate(train_student_x)
    student2.train(
        train_student_x, train_student_y,
        epochs=num_epochs,
        batch_size=100,
        optimizer=optax.sgd(learning_rate=0.5),
        seed=random.randint(0, int(1e9)),
        batches=10
        # return_score=True,
        # evaluate=(test_x, test_y)
    )

    noise = jax.random.uniform(key, shape=(10000, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))
    test_teacher_y = teacher.evaluate(noise)

    acc_student = student.accuracy(noise, test_teacher_y)
    print("Matching student1 (live student) to teacher: {}%".format(acc_student))

    acc_student2 = student2.accuracy(noise, test_teacher_y)
    print("Matching student2 (after student) to teacher: {}%".format(acc_student2))

    acc_student2_vs_student1 = (acc_student2-acc_student)
    print("Accuracy Second Student is {}% more accurate than the first Student".format(acc_student2_vs_student1))

    plt.plot(student_accs, label='Live student')
    plt.plot(latestudents_accs, label='Bright student')
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.legend()
    # plt.axhline(y=acc_student, color='r')
    plt.grid()
    plt.show()
