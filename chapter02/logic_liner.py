import random
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from matplotlib import pyplot as plt

batch_size = 32
lr = 1e-3
w_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])

class LogisticNet(nn.Module):
    def __init__(self):
        super(LogisticNet, self).__init__()
        self.ln = nn.Linear(3, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        out = self.ln(x)
        out = self.sm(out)
        return out

def make_features(x):
    # convert x size to (shape, 1)
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)

def f():
    # Random 0 or 1
    values = []
    for i in range(batch_size):
        data = random.randint(0, 1)
        values.append(float(data))
    #values = np.array(values)
    return values

def get_batch():
    # random create 32 tensor
    random = torch.randn(batch_size)
    x = make_features(random)
    y = torch.FloatTensor(f())
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y)

if __name__ == '__main__':

    # Get Data
    x, y = get_batch()

    if torch.cuda.is_available():
        model = LogisticNet().cuda()
    else:
        model = LogisticNet()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(100):
        # forward pass
        output = model(x)

        loss = criterion(output, y)
        loss_data = loss.data[0]

        # rest gradients
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        optimizer.step()

        # print loss
        print('loss: {:.6f}'.format(loss_data))

    # predict
    model.eval()
    predict_y = model(x)
    predict_y = predict_y.data.numpy()
