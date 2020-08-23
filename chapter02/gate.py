# w1x1+w2x2 <= theta --> y=0
# w1x1+w2x2 > theta --> y=1
def AND_basic(x1, x2):
    w1, w2, theta= 0.5, 0.5, 0.7
    tmp= x1*w1+ x2*w2
    if tmp<= theta:
        return 0
    elif tmp > theta:
        return 1

# AND, NAND, OR은 모두 같은 구조의 퍼셉트론이고, 차이는 가중치 매개변수의 값 뿐이다.

# numpy를 사용하여 구현
# theta= -b
# b(편향) + w1x1+ w2x2 <=0 --> y=0
# b(편향) + w1x1+ w2x2 >0 --> y=1
import numpy as np

def AND(x1, x2):
    x= np.array([x1, x2])
    w= np.array([0.5, 0.5])
    b=-0.7
    tmp= np.sum(w*x)+b
    if tmp <= 0: return 0
    else : return 1

print(
    "AND :",
    AND(0, 0),
    AND(1, 0),
    AND(0, 1),
    AND(1, 1)
)

def NAND(x1, x2):
    x= np.array([x1, x2])
    w= np.array([-0.5, -0.5]) # AND와 가중치, 편향이 다르다.
    b= 0.7
    tmp= np.sum(w*x)+b
    if tmp <= 0: return 0
    else : return 1

print(
    "NAND :",
    NAND(0, 0),
    NAND(1, 0),
    NAND(0, 1),
    NAND(1, 1)
)

def OR(x1, x2):
    x=np.array([x1, x2])
    w= np.array([0.5, 0.5])
    b=-0.2
    tmp= np.sum(w*x)+b
    if tmp <=0: return 0
    else: return 1

print(
    "OR :",
    OR(0, 0),
    OR(1, 0),
    OR(0, 1),
    OR(1, 1)
)

# 앞에서 사용했던 것과 같이 단층 퍼셉트론으로 XOR을 구현하는 것은 불가능하다
# 다층 퍼셉트론을 이용해 구현이 가능하다.

def XOR(x1, x2):
    s1= NAND(x1, x2)
    s2= OR(x1, x2)
    y= AND(s1, s2)
    return y

print(
    "XOR :",
    XOR(0, 0),
    XOR(1, 0),
    XOR(0, 1),
    XOR(1, 1)
)
