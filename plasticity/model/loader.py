import jax
from ml_datasets import mnist

def load_mnist():
    (train_x, train_y), (test_x, test_y) = mnist()

    train_data = [
        (train_x[i].reshape(-1, 1), train_y[i].reshape(-1, 1))
        for i in range(train_x.shape[0])
    ]

    test_data = [
        (test_x[i].reshape(-1, 1), test_y[i].reshape(-1, 1))
        for i in range(test_x.shape[0])
    ]

    return train_data, test_data

def load_mnist_raw():
    return mnist()


def load_mnist_teacher_student():
    (train_x, train_y), (test_x, test_y) = mnist()
    train_teacher_x = train_x[:30000]
    train_teacher_y = train_y[:30000]
    train_student_x = train_x[30000:]
    train_student_y = train_y[30000:]
  
    return (train_teacher_x,train_teacher_y), (train_student_x, train_student_y),(test_x,test_y)