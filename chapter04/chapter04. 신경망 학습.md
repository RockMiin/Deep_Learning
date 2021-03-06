## chapter04. 신경망 학습

기계학습 문제는 데이터를 **훈련 데이터**와 **시험 데이터**로 나눠 실험을 수행한다. 훈련 데이터만 사용하여 학습하면서 최적의 매개변수를 찾는다. 그런 다음 시험 데이터를 사용하여 앞서 훈련한 모델의 실력을 평가한다. 범용적으로 사용할 수 있는 모델을 찾고 제대로 평가하기 위해 나눈다. 한 데이터셋에만 지나치게 최적화 된 상태를 **오버 피팅**이라고 한다.

신경망 학습에서는 현재의 상태를 '하나의 지표'로 표현한다. 이 지표를 가장 좋게 만들어주는 가중치 매개변수의 값을 탐색하는 것이다. 신경망에서도 '하나의 지표'를 기준으로 최적의 매개변수 값을 탐색하는데, 이 때 사용하는 지표를 **손실 함수**라고 한다. 손실 함수는 일반적으로 **평균 제곱 오차**와 **교차 엔트로피 오차**를 사용한다

**원-핫 인코딩** : 한 원소만 1로 하고 그 외는 0으로 나타내는 표기법 

평균 제곱 오차 기준으로 손실 함수 출력이 작은 것이 정답에 가깝다고 말할 수 있다. (mean_squared_error)

**미니 배치 학습**

기계학습 문제는 훈련 데이터를 사용해 학습한다. 훈련 데이터에 대한 손실 함수의 값을 구하고, 그 값을 최대한 줄여주는 매개변수를 찾아낸다. 이러한 손실 함수의 합을 데이터의 개수로 나눔으로써 **평균 손실함수**를 구할 수 있다. 그런데 데이터의 수가  많이지면 많아질수록 모든 데이터를 일일이 손실 함수 계산하는 것은 현실적이지 않기 때문에 일부 데이터를 추려 근사치로 이용할 수 있다. 신경망 학습에서도 훈련 데이터로부터 일부만 골라 학습을 수행하는 **미니 배치 학습**을 한다.

```python
def cross_entropy_error(y, t):
    if y.ndim==1:
        t= t.reshape(1, t.size)
        y= y.reshape(1, y.size)
    batch_size= y.shape[0]
    return -np.sum(t* np.log(y) / batch_size)
```

원-핫 인코딩이 아니라 '2'나 '7' 등의 숫자 레이블로 주어졌을 때의 교차 엔트로피 오차 구현

```python
def cross_entropy_error(y, t):
    if y.ndim==1:
        t= t.reshape(1, t.size)
        y= y.reshape(1, y.size)
    batch_size= y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])/ batch_size)
```

손실함수 사용하는 이유

숫자 인식의 경우도 궁극적인 목적은 높은 '정확도'를 끌어내는 매개변수 값을 찾는 것이다. '정확도'라는 지표를 놔두고 '손실 함수 값'이라는 우회적인 방법을 택하는 이유가 뭘까 ? 

신경망 학습에서의 '미분'의 역할에 주목하면 해결이 된다. 신경망 학습에서는 최적의 매개변수 (가중치와 편향)를 탐색할 때 손실함수의 값을 가능한 작게 하는 매개변수 값을 찾는다. 이 때 매개변수의 미분을 계산하고, 그 미분 값을 단서로 매개변수의 값을 서서히 갱신하는 과정을 반복한다. 가중치 매개변수의 손실함수의 미분이란 **'가중치 매개변수의 값을 아주 조금 변화시켰을 때, 손실 함수가 어떻게 변하나'**의 의미이다. 만약 미분 값이 음수면 그 가중치 매개변수를 양의 방향으로 변화시켜 손실 함수의 값을 줄일 수 있다. **정확도를 지표로 삼아서는 안 되는 이유는 미분 값이 대부분의 장소에서 0ㅣ 되어 매개변수를 갱신할 수 없기 때문이다.** 정확도를 지표로 하는 경우 매개변수를 조정했을 때 불연속적인 띄엄띄엄한 값으로 바뀌기지만 손실 함수 값을 지표로 사용할 경우 연속적으로 변화하기 떄문에 사용한다. '계단 함수'를 활성화 함수로 사용하지 않는 이유와도 비슷하다고 볼 수 있다. 즉, **매개변수의 작은 변화가 주는 파장을 정확도를 지표로 하게 된다면 의미가 없어지기 때문에 손실함수를 지표로 사용한다.**

수치 미분 구현

```python
def numerical_diff(f, x):
    h= 10e-50
    return (f(x+h)-f(x)/h)
```

위와 같이 구현하면 문제점이 2개가 생긴다.

1. h값이 너무 작아 파이썬에서 반올림을 하면 0으로 인식하는 문제
2. h를 무한히 0으로 좁히는 것은 불가능하기에 진정한 미분과는 값이 일치하지 않는다

이 문제점을 개선하여 구현한 코드

```python
def numerical_diff(f, x):
    h= 1e-4
    return (f(x+h)+f(x-h)/(2*h))
```





