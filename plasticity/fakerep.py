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



if __name__ == '__main__':
  
    eras = 30
    student_epochs = 15
    student_final_epochs = eras*student_epochs
    noise_amount_step = 40000
    batch_size = 250

    key = jax.random.PRNGKey(69420)

    model_teacher = presets.Resnet1_mnist(key)
    model_student_along = presets.Resnet1_mnist(key)
    model_student_final = presets.Resnet1_mnist(key)
    optimizer = optax.sgd(learning_rate=0.1)
    opt_state=optimizer.init(model_student_along.params)
    train_data, test_data = loader.load_mnist_raw()


    train_teacher_x, train_teacher_y = train_data
    # train_student_x, _ = train_student
    test_x, test_y = test_data
    random_noise = jax.random.uniform(key, shape=(noise_amount_step * eras, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))

    key2 = jax.random.PRNGKey(69)
    random_noise_test = jax.random.uniform(key2, shape=(40000, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))

    live_student = []
    bright_student = []

    for era in range(eras):
        print("Teacher epochs {}/{}".format(era+1, eras))

        model_teacher.train(
            train_teacher_x, train_teacher_y,
            epochs=1, batch_size=batch_size,
            optimizer=optax.sgd(learning_rate=0.1),
            return_score=False,
            key=jax.random.PRNGKey(random.randint(0, int(1e7))),
            cost=squaredmean_cost,
            # gamma=1,
            # p_slow=0
        )
      
       
        random_noise_step_live = random_noise[(era)*noise_amount_step:((era)+1)*noise_amount_step]
        train_student_y_live = model_teacher.forward(model_teacher.params, random_noise_step_live)
        opt_state = model_student_along.train(
            random_noise_step_live, train_student_y_live,
            epochs=student_epochs, batch_size=batch_size,
            optimizer = optimizer,
            l2=False,
            l2_eps=1e-6,
            opt_state=opt_state,
            key=jax.random.PRNGKey(random.randint(0, int(1e7))
        ))
        brightschtudent = presets.Resnet1_mnist(key)
        opt_state_bright=optimizer.init(brightschtudent.params)
        for i in range(era+1):
            print("Live student epochs:")
            random_noise_step = random_noise[(i)*noise_amount_step:((i)+1)*noise_amount_step]
            print(random_noise_step.device)
            train_student_y = model_teacher.forward(model_teacher.params, random_noise_step)
            opt_state_bright = brightschtudent.train(
                random_noise_step, train_student_y,
                epochs=student_epochs, batch_size=batch_size,
                optimizer = optimizer,
                l2=False,
                l2_eps=1e-6,
                opt_state=opt_state_bright,
                key=jax.random.PRNGKey(random.randint(0, int(1e7))
            ))
       

        teacher_data = model_teacher.forward(model_teacher.params, random_noise_test)

        
        bright_student_data = brightschtudent.forward(brightschtudent.params, random_noise_test)
        live_student_data = model_student_along.forward(model_student_along.params,random_noise_test)
        div_stud_along_teacher_bright = kl_divergence(q=bright_student_data, p=teacher_data)
        bright_student.append(div_stud_along_teacher_bright)
        div_stud_along_teacher_live = kl_divergence(q=live_student_data, p=teacher_data)
        live_student.append(div_stud_along_teacher_live)

print([float(x) for x in bright_student])
print([float(x) for x in live_student])