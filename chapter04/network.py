import sys, os
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient
sys.path.append(os.pardir)

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        # 결국에는 총 3층짜리 신경망이다.
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 신경망을 구현하여서 출력값을 return 해옴
    def predict(self, x):
        # 클래스에 params에서 가중치와 편향 값을 변수로 받아옴
        W1, W2= self.params['W1'], self.params['W2']
        b1, b2= self.params['b1'], self.params['b2']

        a1= np.dot(x, W1) +b1
        z1= sigmoid(a1)
        a2= np.dot(z1, W2) +b2
        y= softmax(a2)

        return y

    # x: 입력 데이터, t: 정답 레이블
    # 입력을 받아와서 학습된 모델을 통해 예측을 한 뒤 손실 함수 값을 return
    def loss(self, x, t):
        y= self.predict(x)

        return cross_entropy_error(y, t)

    # y와 t값을 비교하여 일치하는 개수를 전체 데이터 크기로 나눈 accuracy return
    def accuracy(self, x, t):
        y= self.predict(x)
        y= np.argmax(y, axis=1) # 1차원 확인하기
        t= np.argmax(t, axis=1)

        accuracy= np.sum(y==t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


