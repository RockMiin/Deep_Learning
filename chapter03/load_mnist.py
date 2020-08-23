import sys, os
sys.path.append(os.pardir)
from mnist import load_mnist

(x_train, t_train), (x_test, t_test)= load_mnist(flatten=True,
                                                 normalize=False)
# 인자로는 normalize, flatten, one_hot_label 3가지를 설정할 수 있다.
# normalize : 입력 이미지의 픽셀 값을 정규화 할것인지? (false면 0~255 유지)
# flatten : 입력 이미지를 평탄하게, 1차원 배열로 만들 것인지를 정한다.
# 원-핫 인코딩 : 정답을 뜻하는 원소만 1이고 나머지는 모두 0인 배열로 만들어줌

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

