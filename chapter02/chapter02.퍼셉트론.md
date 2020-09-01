## chapter02. 퍼셉트론

**퍼셉트론**

퍼셉트론은 다수의 신호(흐름)를 입력으로 받아 하나의 신호를 출력한다. 이 신호는 흐른다(1) / 안 흐른다(0)의 두 가지 값만 가질 수 있다.

퍼셉트론은 복수의 각각의 입력 신호에 고유한 가중치를 부여한다. 가중치는 각 신호가 결과에 주는 영향력을 조절하는 요소로 작용한다.

우리는 퍼셉트론을 활용하여 간단한 논리 회로를 구성할 수 있다.

**AND게이트**

```python
def AND_basic(x1, x2):
    w1, w2, theta= 0.5, 0.5, 0.7
    tmp= x1*w1+ x2*w2
    if tmp<= theta:
        return 0
    elif tmp > theta:
        return 1
```

임계값 theta를 -b라는 편향으로 치환

```python
import numpy as np

def AND(x1, x2):
    x= np.array([x1, x2])
    w= np.array([0.5, 0.5])
    b=-0.7
    tmp= np.sum(w*x)+b
    if tmp <= 0: return 0
    else : return 1
```

**NAND게이트**

```python
def NAND(x1, x2):
    x= np.array([x1, x2])
    w= np.array([-0.5, -0.5]) # AND와 가중치, 편향이 다르다.
    b= 0.7
    tmp= np.sum(w*x)+b
    if tmp <= 0: return 0
    else : return 1
```

**OR게이트**

```python
def OR(x1, x2):
    x=np.array([x1, x2])
    w= np.array([0.5, 0.5])
    b=-0.2
    tmp= np.sum(w*x)+b
    if tmp <=0: return 0
    else: return 1
```

지금 까지 본 AND, NAND, OR은 모두 같은 구조의 퍼셉트론이고, **차이는 가중치 매개변수의 값** 뿐이다.

지금 구현해 볼 XOR 게이트는 앞에서 구현한 게이트의 구조로는 구현할 수 없다.

앞에서 구현한 퍼셉트론은  선형 영역으로 나눌 수 있지만, XOR은 비선형 영역으로 나눌 수가 있다. XOR 게이트는 AND, NAND, OR 게이트를 조합하면 완성할 수 있다.

```python
def XOR(x1, x2):
    s1= NAND(x1, x2)
    s2= OR(x1, x2)
    y= AND(s1, s2)
    return y
```



**정리**

퍼셉트론은 다수의 신호를 입력 받아 1이나 0 값으로 출력을 한다.

퍼셉트론에서는 '가중치'와 '편향'을 매개변수로 설정한다.

퍼셉트론으로 AND, OR 게이트 등의 논리 회로를 표현할 수 있다.

단층 퍼셉트론은 직선형 영역만 표현할 수 있고, 다층 퍼셉트론은 비선형 영역도 표현할 수 있다.



