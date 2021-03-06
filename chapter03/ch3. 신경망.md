## 신경망

신경망은 입력층, 은닉층, 출력층 순서로 구성되어 있다.

활성화 함수 : 입력 신호의 총합이 활성화를 일으키는지를 정하는 역할

활성화 함수에는 계단 함수, 시그모이드 함수, ReLU 함수 등이 있다

계단 함수에 비해 시그모이드 함수는 출력이 **연속적으로 변화**한다 -> 매끈함 -> 중요한 역할

공통점은 둘 다 입력이 작을 때 출력은 0에 가깝고, 입력이 커지면 출력이 1에 가까워지는 구조 -> 즉 **입력이 중요하면 큰 값을 출력하고, 입력이 중요하지 않으면 작은 값을 출력**한다 + 입력이 아무리 작거나 커도 **출력은 0과 1사이**이다. + 둘 다 비선형 함수

**신경망에서는 활성화 함수를 비선형 함수로 사용**해야 한다.

​	이유 :  선형 함수를 이용하면 신경망의 층을 깊게하는 의미가 없어짐 -> 층을 아무리 깊게 해도 은	닉층이 없는 네트워크로도 똑같은 기능을 할 수 있다.

​	h(x)= cx라고 가정을 하면 y(x)= h(h(h(x))) = c* c* c* x로 변경 가능 -> 여러 층으로 구성하는 이점을 	살릴 수가 없다.

**ReLU 함수**

​	현재 신경망 분야에서는 ReLU함수를 제일 많이 사용한다. ReLU는 입력이 0이 넘으면 입력을 그	대로 출력하고, 0 이하면 0을 출력



다차원 배열을 내적을 구해주는 np.dot 함수를 이용하면 쉽게 내적을 구할 수 있다 <- 중요 / + 내적	을 할 때 차원을 고려하는 것이 중요함

```python
A1= np.dot(X, W1) + B1
```

이런 식으로 간단하게 표현할 수 있다.

편향과 가중치는 출력층이 되는 노드의 개수만큼으로 정해준다.



신경망은 분류와 회귀 모두에 이용할 수 있다. 다만 둘 중 어떤 문제냐에 따라 출력층에서 사용하느 활성화 함수가 달라진다. 일반적으로 회귀에는 항등 함수, 분류에는 소프트맥스 함수를 사용

분류 : 데이터가 어느 클래스에 속하느냐는 문제																				 회귀 : 입력 데이터에서 (연속적인) 수치를 예측하는 문제



항등 함수와 다르게 소프트맥스 함수의 출력은 모든 입력 신호로부터 영향을 받는다.



소프트맥스 함수 구현 시 주의할 점

```python
def softmax(a):
    exp_a= np.exp(a)
    sum_exp_a= np.sum(exp_a)
    y= exp_a/sum_exp_a
    
    return y
```

컴퓨터로 계산할 때는 소프트맥스 함수가 지수 함수를 사용하기 때문에 큰 값이 발생해 오버플로 문제가 발생한다. 큰 값 끼리 나눗셈을 하면 결과 수치가 불안정해진다.

소프트맥스 함수 식에 임의의 정수를 곱해서 식을 풀어보면 **소프트맥스의 지수 함수를 계산할 때 어떤 정수를 더해도 결과가 바뀌지 않는다는 것**을 볼 수 있다. 이 C값에 어떤 값을 대입해도 상관이 없지만 **오버플로를 막을 목적으로는 입력 신호 중 최대값을 이용**하는 것이 일반적이다. 그래서 구현한 함수는 다음과 같다

```python
def softmax(a):
	c= np.max(a) # 오버플로 대책
    exp_a= np.exp(a- c)
    sum_exp_a= np.sum(exp_a)
    y= exp_a/sum_exp_a
    
    return y
```



**소프트맥스 함수의 특징**

​	소프트맥스 함수의 출력은 0에서 1.0 사이의 실수

​	출력의 총 합이 1 -> **확률로 해석 가능**

​	소프트맥스 함수를 적용해도 각 원소의 대소 관계는 변하지 않는다 - > 지수 함수가 단조 증가함수	이기 떄문 / a의 원소 사이의 대소 관계가 y의 원소 사이의 대소 관계로 그대로 이어짐 

​	-> 신경망을 이용한 분류에서는 일반적으로 가장 큰 출력을 내는 뉴런에 해당하는 클래스로 인식 	소프트맥스 함수를 적용해도 출력이 가장 큰 뉴런의 위치는 달라지지 않는다 - > 결과적으로 신경	망으로 분류할 떄는 출력층의 소프트맥스 함수를 생략해도 된다.



**손글씨 숫자 인식**

​	**MNIST**

​		0~9까지의 숫자 이미지로 구성

​		trainset 60000장, testset 10000장으로 구성

​		28*28 크기의 회색조(1채널)로 구성, 픽셀 값은 0~255 값을 취함.

​		label이 그 이미지가 실제 의미하는 숫자로 구성

​	**전처리** : 신경망의 입력 데이터에 특정 변환을 가하는 것

​		정규화(normalization) : 0~255 범위인 각 픽셀 값을 0.0 ~1.0 범위로 변환

​		이번 예제에서는 각 픽셀 값을 255로 나누는 단순한 정규화를 하였지만 현업에서는 데이터 전체		의 분포를 고려해 전처리를 함 

​		ex) 데이터 전체 평균과 표준 편차를 이용하여 데이터들이 0을 중심으로 분포하도록 이동하거나 		데이터의 확산 범위를 제한, 전체 데이터를 균일하게 분포시키는 데이터 백색화(whitening) 등이 		있다.

​	**배치 처리**

​		구현을 하기 앞서 입력 데이터와 가중치 매개변수의 '형상'에 항상 주의하여 구현해야 한다.

​		x.shape와 w.shape가 각 층마다 맞물려서 구성되어 있는지 확인하기

​		x= 100장의 28*28의 이미지

​		x				w1			 w2			w3 		  -> 	y

​		100x784	784x50 	50x100 	100x10	->     100x10

​		배치 처리는 컴퓨터로 계산할 때 이점을 준다 -> 이미지 1장당 처리 시간을 대폭 줄여줌

​			이유 1. 수치 계산 라이브러리 대부분 큰 배열을  효율적으로 처리할 수 있도록 구성

​			이유 2. 배치 처리를 함으로써 데이터 전송 BUS의 부하를 줄여줌 (느린 I/O를 통해 데이터를 			읽는 횟수가 줄어서)



이번 장에서는 신경망의 순전파에 대해 공부를 했다. 이번 장에서 설명한 신경망은 각 층의 뉴런들이 다음 층의 뉴런으로 신호를 전달한다는 점에서 앞장의 퍼셈트론과 같지만 다음 뉴런으로 갈 때 신호를 변화시키는 활성화 함수에 큰 차이가 있었다. 