import numpy as np
from dataset.mnist import load_mnist
from network import TwoLayerNet


(x_train, y_train), (x_test, y_test)= load_mnist(normalize=True, one_hot_label=True)

# 손실함수값을 담는 리스트
train_loss_list= []
train_acc_list= []
test_acc_list= []

# 하이퍼 파라미터
iters_num= 100 # 반복 횟수
train_size= x_train[0].size # 784
batch_size= 10
learning_rate= 0.1

iter_per_epoch= max(train_size/ batch_size, 1)

# 신경망
network= TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
print("learning start")
for i in range(iters_num):
    print('start')
    # 미니 배치 값 추출
    batch_mask= np.random.choice(train_size, batch_size)
    x_batch= x_train[batch_mask]
    y_batch= y_train[batch_mask]

    grad= network.numerical_gradient(x_batch, y_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -=learning_rate*grad[key]

    loss= network.loss(x_batch, y_batch)
    print('epoch :', i, 'loss :', loss)
    train_loss_list.append(loss)

    if i% iter_per_epoch== 0:
        train_acc= network.accuracy(x_train, y_train)
        test_acc= network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("train acc:", str(train_acc), "test acc:", str(test_acc))
    print('end')