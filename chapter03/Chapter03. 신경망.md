## Chapter03. 신경망

컴퓨터가 수행하는 복잡한 처리도 퍼셉트론으로 이론상 표현할 수 있다. 하지만 **가중치를 설정하는 작업은 여전히 사람이 수동**으로 한다.

신경망은 **가중치 매개변수의 적절한 값을 데이터로부터 자동으로 학습하는 능력**을 가지고 있다.

**신경망**

신경망은 퍼셉트론과 공통점이 많지만 이번 장에서는 다른 점을 중심으로 신경망의 구조를 설명한다.

신경망은 입력층, 은닉층, 출력층으로 구성되어 있다.

활성화 함수는 임계값을 경계로 출력이 바뀌는데, 이러한 함수를 **계단함수**라고 한다. 그래서 퍼셉트론에서는 활성화 함수로 계단 함수를 이용한다라고 할 수 있다. 활성화 함수를 계단 함수 이외의 함수를 사용하는 것이 신경망의 세계를 나아가는 열쇠이다.

**계단함수**

```python
def step_function(x):
    if x>0:
        return 1
    else:
        return 0
```

위 구현은 인수 x는 실수만 받아들인다는 단점이 있기 때문에 numpy 배열도 지원하도록 구현해보자.

```python
import numpy as np
def step_function(x):
    y=x>0
    return y.astype(np.int)
```

계단 함수의 그래프

```python
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0, dtype=np.int)

x=np.arange(-5.0, 5.0, 0.1)
y= step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
```



**시그모이드 함수**

h(x)= 1/ (1+exp(-x))

얼핏 복잡해 보이지만 이 역시 단순한 '함수'이다. 입력을 주면 출력을 돌려주는 변환기이다.

```python
def sigmoid(x):
    return 1/ (1+np.exp(-x))

x= np.arange(-5.0, 5.0, 0.1)
y=sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
```



**시그모이드 함수와 계단 함수 비교**

시그모이드 함수 :  부드러운 곡선이며 입력에 따라 출력이 **연속적**으로 변함

계단 함수 : 0을 경계로 출력이 갑자기 바뀌어림

하지만 두 함수 모두 입력이 작을 때의 출력은 0에 가깝고 입력이 커지면 출력이 1에 가까워지는 구조이다. 또한 입력에 상관없이 출력은 0에서 1 사이이다. 중요한 공통점은, 두 함수 모두 **비선형 함수**이다.



신경망에서는 활성화 함수로 비선형 함수를 사용해야 한다. 왜냐하면 선형 함수를 이용하면 신경망의 층을 깊게 하는 의미가 없기 때문이다. 층을 쌓는 혜택을 얻고 싶다면 활성화 함수로는 반드시 비선형 함수를 사용해야 한다.



**ReLU 함수**

```python
def relu(x):
    return np.maximum(0, x)
```

**미니 배치 학습**

기계학습 문제는 훈련 데이터를 사용해 학습한다. 훈련 데이터에 대한 손실 함수의 값을 구하고, 그 값을 최대한 줄여주는 매개변수를 찾아낸다. 이러한 손실 함수의 합을 데이터의 개수로 나눔으로써 **평균 손실함수**를 구할 수 있다. 그런데 데이터의 수가  많이지면 많아질수록 모든 데이터를 일일이 손실 함수 계산하는 것은 현실적이지 않기 때문에 일부 데이터를 추려 근사치로 이용할 수 있다. 신경망 학습에서도 훈련 데이터로부터 일부만 골라 학습을 수행하는 **미니 배치 학습**을 한다.

**전처리** : 신경망의 입력 데이터에 특정 변환을 가하는 것

**정규화** : 데이터를 특정 범위로 변한하는 처리

**배치**는 간단하게 말하면 묶음이라고 생각
