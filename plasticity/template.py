import math
import jax
import optax
import random
import matplotlib.pyplot as plt

import loader
import presets
from model import *
from model import _gen_loss_function

# DO NOT CHANGE TEACHER. I TUNED FOR THE BEST POSSIBLE TEACHER (Alessandro Farcaş)
# adamw lr: 0.0005; wd: 0.0001; bs: 125
# Accuracy of teacher on test data: 97.98%
# Accuracy of student on training data: 99.73%

if __name__ == '__main__':
    seed = random.randint(0, int(1e9))
    key = jax.random.PRNGKey(seed)
    models_seeds, test_noise_seed, key = jax.random.split(key, 3)

    train_data, test_data = loader.load_mnist_raw()
    train_x, train_y = train_data
    test_x, test_y = test_data

    teacher_eras = 30
    teacher_lr = 0.0005
    teacher_wd = 0.0001
    teacher_bs = 125
    model_teacher = presets.Resnet1_mnist(models_seeds)
    optimizer_teacher = optax.adamw(learning_rate=0.0005, weight_decay=0.0001)
    optimizer_teacher_state = optimizer_teacher.init(model_teacher.params)

    model_student_live = presets.Resnet1_mnist(models_seeds)
    model_student_final = presets.Resnet1_mnist(models_seeds)

    loss_fn_teacher = _gen_loss_function(model_teacher.forward, crossentropy_cost)

    student_epochs = 15
    bright_student_epochs = teacher_eras * student_epochs
    batch_size_student = 100

    test_noise_amount = 10000
    train_noise_amount = 80000

    test_noise = jax.random.uniform(test_noise_seed, shape=(test_noise_amount, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))

    optimizer_student = optax.sgd(learning_rate=0.2, momentum=0.8)
    optimizer_student_state = optimizer_student.init(model_student_live.params)

    loss_fn_student = _gen_loss_function(model_student_live.forward, crossentropy_cost)

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

        for epoch in range(student_epochs):
            print("Epoch: {}/{}".format(epoch+1, student_epochs))
            key, noise_key = jax.random.split(key)
            train_noise = jax.random.uniform(noise_key, (train_noise_amount, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))
            train_noise_y = model_teacher.evaluate(train_noise)

            key, student_train_key = jax.random.split(key)
            print('start training')
            model_student_live.params, optimizer_student_state, losses = train_epoch(
                params=model_student_live.params,
                opt_state=optimizer_student_state,
                x=train_noise,
                y=train_noise_y,
                optimizer=optimizer_student,
                loss_fn=loss_fn_student,
                batches=train_noise_amount//batch_size_student,
                batch_size=batch_size_student,
                key=student_train_key
            )
            print(jnp.mean(losses))
            print('end training')
        
    acc_teacher_train = model_teacher.accuracy(train_x, train_y)
    acc_teacher_test = model_teacher.accuracy(test_x, test_y)
    print(acc_teacher_train)
    print(acc_teacher_test)

    acc_student_train = model_student_live.accuracy(train_x, train_y)
    acc_student_test = model_student_live.accuracy(test_x, test_y)
    print(acc_student_train)
    print(acc_student_test)