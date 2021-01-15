import numpy as np
import sys, os
from dataset.mnist import load_mnist

# y는 예측 값, test는 정답 label
def mean_squared_error(y, test):
    return 0.5* np.sum((y-test)**2)

def cross_entropy_error(y, test):
    if y.ndim== 1:
        test= test.reshape(1, test.size)
        y= y.reshape(1, y.size)
    batch_size= y.shape[0]
    return -np.sum(test *np.log(y))/ batch_size


(x_train, y_train), (x_test, y_test)= load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)
print(y_train.shape)

train_size= x_train.shape[0]
batch_size= 10
batch_mask= np.random.choice(train_size, batch_size) # train_size 만큼의 정수에서 batch_size 개수만큼 추출
print(batch_mask)
x_batch, y_batch= x_train[batch_mask], y_train[batch_mask]

cross_entropy_error()