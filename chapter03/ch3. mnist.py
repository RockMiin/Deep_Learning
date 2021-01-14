import sys, os
from dataset.mnist import load_mnist
import pickle
import numpy as np

# (x_train, y_train), (x_test, y_test)= load_mnist(flatten=True, normalize=False)
#
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# 이미 sample로 만들어 놓은 가중치 셋을 가지고 예측을 하는 것
def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=True,
                                                      normalize=True, one_hot_label=False)
    return x_test, y_test

def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network= pickle.load(f)

    return network

# 활성화 함수
def sigmoid(x):
    return 1/ (1+ np.exp(-x))


def softmax(a):
    c = np.max(a)  # 오버플로 대책
    
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

def predict(network, x):
    W1, W2, W3= network['W1'], network['W2'], network['W3']
    b1, b2, b3= network['b1'], network['b2'], network['b3']

    a1= np.dot(x, W1)+ b1
    z1= sigmoid(a1)
    a2= np.dot(z1, W2)+ b2
    z2= sigmoid(a2)
    a3= np.dot(z2, W3)+ b3
    y= softmax(a3)

    return y

# test_x, test_y= get_data()
# network= init_network()
#
# acc= 0
# for i in range(len(test_x)):
#     y= predict(network, test_x[i])
#     pred_y= np.argmax(y)
#     if pred_y==test_y[i]:
#         acc+=1
# print("accuracy:" +str(float(acc/len(test_x))))

# batch
test_x, test_y= get_data()
network= init_network()

batch_size= 100
acc= 0

for i in range(0, len(test_x), batch_size):
    batch_x= test_x[i:i+batch_size]
    batch_y= predict(network, batch_x)
    p= np.argmax(batch_y, axis=1)
    acc+= np.sum(p== test_y[i:i+batch_size])

print("accuracy:" +str(float(acc/len(test_x))))



