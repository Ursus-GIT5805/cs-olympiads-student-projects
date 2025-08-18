# import math
# import jax
# import optax
# import loader
# from linear import *
# from model import Model
# from model import batch_norm
# from model import kl_divergence
# from model import squaredmean_cost
# import matplotlib.pyplot as plt
# import copy
# import random

# def create_model(params):
#     def run(params, a):
#         a = feedforward_linear(params[0], a)

#         x1 = a.copy()

#         a = jax.nn.sigmoid(a)
#         a = feedforward_linear(params[1], a)
#         a = batch_norm(a)
#         a = jax.nn.relu(a)

#         a = feedforward_linear(params[2], a)
#         a = batch_norm(a)

#         a = a + x1
#         a = jax.nn.relu(a)

#         a = feedforward_linear(params[3], a)
#         a = jax.nn.softmax(a)
#         return a

#     return Model.init(
#         params,
#         jax.jit(run),
#     )

# if __name__ == '__main__':
#     teacher_epochs = 15
#     student_epochs = 30
#     student_final_epochs = teacher_epochs*student_epochs
#     noise_amount_step = 40000
#     batch_size = 250

#     key = jax.random.PRNGKey(69420)
#     params = linears_from_array([784, 100, 100, 100, 10], key=key)

#     model_teacher = create_model(copy.deepcopy(params))
#     model_student_along = create_model(copy.deepcopy(params))
#     model_student_final = create_model(copy.deepcopy(params))

#     train_teacher, train_student, test_data = loader.load_mnist_teacher_student()

#     train_teacher_x, train_teacher_y = train_teacher
#     train_student_x, _ = train_student
#     test_x, test_y = test_data

#     random_noise = jax.random.uniform(key, shape=(noise_amount_step * teacher_epochs, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))

#     key2 = jax.random.PRNGKey(69)
#     random_noise_test = jax.random.uniform(key2, shape=(40000, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))

#     student_epochs_along_divergence = []
#     accuracies = []

#     for epoch in range(teacher_epochs):
#         print("Teacher epochs {}/{}".format(epoch+1, teacher_epochs))

#         print("Teacher learning")
#         model_teacher.train(
#             train_teacher_x, train_teacher_y,
#             epochs=1, batch_size=batch_size,
#             optimizer=optax.sgd(learning_rate=0.1),
#             return_score=False,
#             cost=squaredmean_cost,
#             # evaluate=(test_x, test_y),
#             seed=random.randint(0, int(1e7))
#         )

#         # the_key = jax.random.PRNGKey(epoch)
#         # random_noise_step = random_noise[epoch*noise_amount_step:(epoch+1)*noise_amount_step]
#         # print(random_noise_step.device)
#         # train_student_y = model_teacher.forward(model_teacher.params, random_noise_step)
#         train_student_y = model_teacher.evaluate(train_student_x)

#         teacher_data = model_teacher.evaluate(random_noise_test)

#         print("Live student epochs:")
#         for student_epoch in range(student_epochs):
#             print("Epoch: {}/{}".format(student_epoch+1, student_epochs))
#             model_student_along.train(
#                 train_student_x, train_student_y,
#                 epochs=1, batch_size=batch_size,
#                 optimizer = optax.sgd(learning_rate=0.1),
#                 #return_score=True,
#                 #evaluate=(test_x, test_y),
#             )
#             along_student_acc = model_student_along.accuracy(test_x, test_y)
#             accuracies.append(along_student_acc/100)

#             along_student_data = model_student_along.forward(model_student_along.params, random_noise_test)
#             div_stud_along_teacher = kl_divergence(q=along_student_data, p=teacher_data)
#             student_epochs_along_divergence.append(div_stud_along_teacher)

        
#     print("After student epochs:")

# #    train_student_y_final = model_teacher.forward(model_teacher.params, random_noise)
# #    model_student_final.train(
# #        random_noise, train_student_y_final,
# #        epochs=student_final_epochs, batch_size=batch_size,
# #        optimizer = optax.sgd(learning_rate=0.1),
# #        return_score=False,
# #        # evaluate=(test_x, test_y),
# #    )
# #
#     acc_train = model_teacher.accuracy(train_teacher_x, train_teacher_y)
#     acc_test = model_teacher.accuracy(test_x, test_y)
#     print("Accuracy teacher on training data: {}%".format(acc_train))
#     print("Accuracy teacher on test data: {}%".format(acc_test))
 
#     acc_train = model_student_along.accuracy(train_student_x, train_student_y)
#     acc_test = model_student_along.accuracy(test_x, test_y)
#     print("Accuracy live student on training data: {}%".format(acc_train))
#     print("Accuracy live student on test data: {}%".format(acc_test))
# #
# #    acc_train = model_student_final.accuracy(train_x, train_y)
# #    acc_test = model_student_final.accuracy(test_x, test_y)
# #    print("Accuracy after student on training data: {}%".format(acc_train))
# #    print("Accuracy after student on test data: {}%".format(acc_test))
# #
# #
# #    teacher_data = model_teacher.forward(model_teacher.params, random_noise_test)
# #    along_student_data = model_student_along.forward(model_student_along.params, random_noise_test)
# #    final_student_data = model_student_final.forward(model_student_final.params, random_noise_test)
# #
# #    div_stud_follow_teacher = kl_divergence(q=along_student_data, p=teacher_data)
# #    div_stud_final_teacher = kl_divergence(q=final_student_data, p=teacher_data)
# #
# #    print("Divergence of live student to teacher: {}".format(div_stud_follow_teacher))
# #    print("Divergence of after student to teacher: {}".format(div_stud_final_teacher))
#     print([float(x) for x in student_epochs_along_divergence])
#     # plt.plot(student_epochs_along_divergence, label='Student')
#     # plt.xlabel("Epochs")
#     # plt.ylabel("KL Divergence")
#     # plt.figtext(0, 0, "KL Divergence between teacher and student  with student having 30 epochs \n for each teacher epoch with optimizer sgd  and learning rate 0.1",fontsize=10)
#     # # plt.figtext(0, 0, "Weight Magnitudefor every student epoch (sgd with 0.2 learning rate)", fontsize = 10)
#     # plt.grid()
#     # plt.clf()
#     # # plt.legend()
#     # # plt.plot(accuracies, label='accuracies')
#     # plt.show()

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
    batch_size = 250
    
    key = jax.random.PRNGKey(69420)
    

    model_teacher = presets.Resnet1_mnist(key)
    model_student_along = presets.Resnet1_mnist(key)
    model_student_final = presets.Resnet1_mnist(key)

    train_data, test_data = loader.load_mnist_raw()


    train_teacher_x, train_teacher_y = train_data
    # train_student_x, _ = train_student
    test_x, test_y = test_data
    random_noise = jax.random.uniform(key, shape=(noise_amount_step * teacher_epochs, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))
    optimizer2=optax.adamw(learning_rate=0.00005, weight_decay=0.1)
    key2 = jax.random.PRNGKey(69)
    random_noise_test = jax.random.uniform(key2, shape=(40000, 784), minval=-math.sqrt(3), maxval=math.sqrt(3))

    student_epochs_along_divergence = []
    accuracies = []
    # model_student_along.resetsubset()
    opt_state2=optimizer2.init(model_student_along.params)
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
        random_noise_step = random_noise[epoch*noise_amount_step:(epoch+1)*noise_amount_step]
        print(random_noise_step.device)
        train_student_y = model_teacher.forward(model_teacher.params, random_noise_step)

        teacher_data = model_teacher.forward(model_teacher.params, random_noise_test)

        print("Live student epochs:")
        # for student_epoch in range(student_epochs):
        # print("Epoch: {}/{}".format(student_epoch+1, student_epochs))
        current_student_epochs = getepochsforstudent(epoch,teacher_epochs,student_epochs,5)
        print(current_student_epochs)
        # model_student_along.model_reset_top(p=0.0001, seed=random.randint(0, int(1e7)))
        opt_state2 = model_student_along.train(
            random_noise_step, train_student_y,
            epochs=student_epochs, batch_size=batch_size,
            optimizer = optimizer2,
            opt_state=opt_state2,
            # gamma=0.9,
            # p_slow=0.
            #return_score=True,
            #evaluate=(test_x, test_y),
        )
        along_student_acc = model_student_along.accuracy(test_x, test_y)
        accuracies.append(along_student_acc/100)

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
