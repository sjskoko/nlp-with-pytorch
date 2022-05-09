import imp
from pickletools import optimize
import torch
import matplotlib.pyplot as plt

'''
본 파일은 "파이토치로 배우는 자연어처리"의 3장 내용을 담고 있으며
1. activation function
2. loss function
3. optimizer
의 내용을 포함합니다
'''

###
# activation function
###

# sigmoid
x = torch.range(-5, 5, 0.1)
y = torch.sigmoid(x)
plt.plot(x.numpy(), y.numpy())
plt.show()

# hyperbolic tangent
x = torch.range(-5, 5, 0.1)
y = torch.tanh(x)
plt.plot(x.numpy(), y.numpy())
plt.show()

# relu
x = torch.range(-5, 5, 0.1)
y = torch.nn.ReLU(x) # nn에 포함되어있음
plt.plot(x.numpy(), y.numpy())
plt.show()

# Prelu
x = torch.range(-5, 5, 0.1)
y = torch.nn.PReLU(x)
plt.plot(x.numpy(), y.numpy())
plt.show()

# softmax
softmax = torch.nn.Softmax(dim=1)
x_input = torch.randn(1, 3)
y_output = softmax(x_input)
print(x_input)  
print(y_output)
print(torch.sum(y_output, dim=1))

    # 참고
    # randn 함수는 input값을 dimention으로 하고 X~normal(0, 1)를 원소로하는 "텐서"를 반환함
    # 텐서를 반환하므로 requires_grad 설정이 가능하다(gradient 계산)



###
# loss function
###

# mse
mse_loss = torch.nn.MSELoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.randn(3, 5)
loss = mse_loss(outputs, targets)
print(loss)

# cross entropy
ce_loss = torch.nn.CrossEntropyLoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.tensor([3, 1, 4], dtype=torch.int64)
loss = ce_loss(outputs, targets)
print(loss)

    # 참고
    # 파이토치의 cross entropy 계산 방식은, 
    # (자료 개수, class 종류의 개수)의 output과, [각 자료의 class 번호]로 이루어진 target값을 토대로 계산된다

# binary cross entropy (BCE)
bce_loss = torch.nn.BCELoss()
sigmoid = torch.nn.Sigmoid()
probabilities = sigmoid(torch.randn(4, 1, requires_grad=True))
targets = torch.tensor([0, 1, 1, 1], dtype=torch.float32).view(4, 1)
loss = bce_loss(probabilities, targets)
print(probabilities)
print(loss)

    # 참고
    # 이진엔트로피 손실의 경우, 각각의 자료가 sigmoid를 통과한 실수값으로 들어가게 되며, 
    # 각각이 0인지 1인지 판별하게 된다


###
# optimizer
###
from Summary.perceptron import *
import numpy as np

LEFT_CENTER = (3, 3)
RIGHT_CENTER = (3, -2)
n_epochs = 1000
n_batchs = 32
batch_size = 32

def get_toy_data(batch_size, left_center=LEFT_CENTER, right_center=RIGHT_CENTER):
    x_data = []
    y_targets = np.zeros(batch_size)
    for batch_i in range(batch_size):
        if np.random.random() > 0.5:
            x_data.append(np.random.normal(loc=left_center))
        else:
            x_data.append(np.random.normal(loc=right_center))
            y_targets[batch_i] = 1
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_targets, dtype=torch.float32)


input_dim = 2
lr =0.001

perceptron = Perceptron(input_dim=input_dim)
bce_loss = torch.nn.BCELoss()
optimizer = torch.optim.AdamW(params=perceptron.parameters(), lr=lr)


for epoch_i in range(n_epochs):
    for batch_i in range(n_batchs):

        # 0. data load
        x_data, y_target = get_toy_data(batch_size)
        y_target = y_target.view(32,1)

        # 1. gradient reset
        perceptron.zero_grad()

        # 2. foward propagation
        y_pred = perceptron(x_data)

        # 3. calculate loss
        loss = bce_loss(y_pred, y_target)

        # 4. backward propagation
        loss.backward()

        # 5. optimize
        optimizer.step()

[i for i in perceptron.parameters()]