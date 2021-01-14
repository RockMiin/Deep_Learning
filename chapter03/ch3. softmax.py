import numpy as np

a= np.array([0.3, 2.9, 4.0])
exp_a= np.exp(a)
sum_exp_a= sum(exp_a)

# print(exp_a.shape, sum_exp_a.shape)
y= exp_a/sum_exp_a # exp_a는 배열 sum_exp_a는 상수인데도 가능
print(y)

def softmax(a):
    exp_a= np.exp(a)
    sum_exp_a= np.sum(exp_a)
    y= exp_a/sum_exp_a

    return y


def softmax(a):
    c = np.max(a)  # 오버플로 대책


exp_a = np.exp(a - c)
sum_exp_a = np.sum(exp_a)
y = exp_a / sum_exp_a

return y