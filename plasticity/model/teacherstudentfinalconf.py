import math
import jax
import optax

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
    teacher_epochs = 30
    student_epochs = 15
    student_final_epochs = teacher_epochs*student_epochs
    noise_amount_step = 40000
    batch_size = 100

    key = jax.random.PRNGKey(69420)
    
    model_teacher = presets.Resnet1_mnist(key)
    model_student_along = presets.Resnet4_mnist(key)
    model_student_final = presets.Resnet4_mnist(key)
    optimizer = optax.sgd(0.1,0.7)
    opt_state=optimizer.init(model_student_along.params)

    train_data, test_data = loader.load_mnist_raw()


    train_teacher_x, train_teacher_y = train_data
    # train_student_x, _ = train_student
    test_x, test_y = test_data
    random_noise = jax.random.uniform(key, shape=(noise_amount_step * teacher_epochs, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))

    key2 = jax.random.PRNGKey(69)
    random_noise_test = jax.random.uniform(key2, shape=(40000, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))

    student_epochs_along_divergence = []
    accuracies = []
    # model_student_along.resetsubset()

    for epoch in range(teacher_epochs):
        print("Teacher epochs {}/{}".format(epoch+1, teacher_epochs))

        print("Teacher learning")
        model_teacher.train(
            train_teacher_x, train_teacher_y,
            epochs=1, batch_size=batch_size,
            optimizer=optax.sgd(learning_rate=0.1),
            return_score=False,
            cost=squaredmean_cost,
            # evaluate=(test_x, test_y),
            seed=random.randint(0, int(1e7)),
            # gamma=1,
            # p_slow=0
        )
        
        # the_key = jax.random.PRNGKey(epoch)
       


        print("Live student epochs:")
        # for student_epoch in range(student_epochs):
        # print("Epoch: {}/{}".format(student_epoch+1, student_epochs))
        # for faketeacherepoch in range(epoch+1):
        # current_student_epochs = getepochsforstudent(epoch,teacher_epochs,student_epochs,5)
        # print(current_student_epochs)
        # model_student_along.model_reset_top(p=0.0001, seed=random.randint(0, int(1e7)))
        random_noise_step = random_noise[epoch*noise_amount_step:(epoch+1)*noise_amount_step]
        print(random_noise_step.device)
        train_student_y = model_teacher.forward(model_teacher.params, random_noise_step)
        opt_state = model_student_along.train(
            random_noise_step, train_student_y,
            epochs=student_epochs, batch_size=batch_size,
            optimizer = optimizer,
            l2=False,
            l2_eps=1e-6,
            opt_state=opt_state
            # gamma=0.9,
            # p_slow=0.
            #return_score=True,
            #evaluate=(test_x, test_y),
        )
        teacher_data = model_teacher.forward(model_teacher.params, random_noise_test)

        along_student_acc = model_student_along.accuracy(test_x, test_y)
        accuracies.append(along_student_acc/100)
        deads = model_student_along.deads(model_student_along.params,random_noise_test)
        print(deads)
        along_student_data = model_student_along.forward(model_student_along.params, random_noise_test)
        div_stud_along_teacher = kl_divergence(q=along_student_data, p=teacher_data)
        student_epochs_along_divergence.append(div_stud_along_teacher)


    print("After student epochs:")

#    train_student_y_final = model_teacher.forward(model_teacher.params, random_noise)
#    model_student_final.train(
#        random_noise, train_student_y_final,
#        epochs=student_final_epochs, batch_size=batch_size,
#        optimizer = optax.sgd(learning_rate=0.1),
#        return_score=False,
#        # evaluate=(test_x, test_y),
#    )
#
    acc_train = model_teacher.accuracy(train_teacher_x, train_teacher_y)
    acc_test = model_teacher.accuracy(test_x, test_y)
    print("Accuracy teacher on training data: {}%".format(acc_train))
    print("Accuracy teacher on test data: {}%".format(acc_test))

    acc_train = model_student_along.accuracy(train_teacher_x, train_teacher_y)
    acc_test = model_student_along.accuracy(test_x, test_y)
    print("Accuracy live student on training data: {}%".format(acc_train))
    print("Accuracy live student on test data: {}%".format(acc_test))
    print([float(x) for x in student_epochs_along_divergence])
#
#    acc_train = model_student_final.accuracy(train_x, train_y)
#    acc_test = model_student_final.accuracy(test_x, test_y)
#    print("Accuracy after student on training data: {}%".format(acc_train))
#    print("Accuracy after student on test data: {}%".format(acc_test))
#
#
#    teacher_data = model_teacher.forward(model_teacher.params, random_noise_test)
#    along_student_data = model_student_along.forward(model_student_along.params, random_noise_test)
#    final_student_data = model_student_final.forward(model_student_final.params, random_noise_test)
#
#    div_stud_follow_teacher = kl_divergence(q=along_student_data, p=teacher_data)
#    div_stud_final_teacher = kl_divergence(q=final_student_data, p=teacher_data)
#
#    print("Divergence of live student to teacher: {}".format(div_stud_follow_teacher))
#    print("Divergence of after student to teacher: {}".format(div_stud_final_teacher))

    # plt.plot(student_epochs_along_divergence, label='divergence')
