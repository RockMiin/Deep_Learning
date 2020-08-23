import numpy as np
import matplotlib.pylab as plt

# 3.2.2 계단 함수 구현
# 주의할 점 : x는 실수만 받아들인다. numpy 배열 불가 ! / np 배열 지원하도록 수정
def step_function(x):
    # if x>0: return 1
    # else: return 0
    y= x>0
    # numpy에서 자료형을 변환할때 astpye()을 사용
    return y.astype(np.int)

x= np.array([-1.0, 1.0, 2.0])
print(step_function(x))

# 3.2.3 계단 함수 그래프
x= np.arange(-5.0, 5.0, 0.1)
y= step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 3.2.4 시그모이드(=S자 모양) 함수 구현
# 넘파이의 브로드캐스트 기능 때문에 처리가 가능
def sigmoid(x):
    return 1 / (1+np.exp(-x))

x= np.arange(-5.0, 5.0, 0.1)
y=sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 3.2.7 ReLU 함수
def relu(x):
    return np.maximum(0, x)

# 3.3.3 신경망의 내적
X= np.array([1, 2])
W= np.array([[1, 3, 5], [2, 4, 6]])
Y= np.dot(X, W)
print(Y)

# 3.4 신경망의 구현
def identity_function(x):
    return x

def init_network(): # 가중치와 편향을 초기화 하고 딕셔너리 변수에 저장
    network={}
    network['W1']= np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1']= np.array([0.1, 0.2, 0.3])
    network['W2']= np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2']= np.array([0.1, 0.2])
    network['W3']= np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3']= np.array([0.1, 0.2])

    return network

def forward(network, x): # 입력 신호를 출력으로 변환하는 처리 과정 모두 구
    W1, W2, W3= network['W1'], network['W2'], network['W3']
    b1, b2 ,b3= network['b1'], network['b2'], network['b3']

    a1= np.dot(x, W1) + b1
    z1= sigmoid(a1)
    a2= np.dot(z1, W2) + b2
    z2= sigmoid(a2)
    a3= np.dot(z2, W3) + b3
    y= identity_function(a3)

    return y

network= init_network()
x= np.array([1.0, 0.5])
y= forward(network, x)
print(y)

# 3.5 항등 함수와 소프트맥스 함수
def softmax(a):
    c= np.max(a)
    exp_a= np.exp(a-c) #오버플로를 방지하기 위해 c을 사용
    sum_exp_a= np.sum(exp_a)
    y= exp_a / sum_exp_a

    return y

a= np.array([0.3, 2.9, 4.0])
y= softmax(a)
print(y)
print(np.sum(y)) # 총합이 1이 된다는 중요한 성질 -> 확률로 해석할 수 있다.

