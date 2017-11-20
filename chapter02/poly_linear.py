import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from matplotlib import pyplot as plt

lr = 1e-3
w_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])

class PolyNet(nn.Module):
    def __init__(self):
        super(PolyNet, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out

def make_features(x):
    # convert x size to (shape, 1)
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)

def f(x):
    # Matrix multiplication
    return x.mm(w_target) + b_target[0]

def get_batch(batch_size=32):
    # random create 32 tensor
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y)

if __name__ == '__main__':

    # Get Data
    batch_x, batch_y = get_batch()

    if torch.cuda.is_available():
        model = PolyNet().cuda()
    else:
        model = PolyNet()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    while True:
        # forward pass
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss_data = loss.data[0]

        # rest gradients
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        optimizer.step()

        if loss_data < lr:
            break

    # predict
    model.eval()
    predict_y = model(batch_x)
    predict_y = predict_y.data.numpy()

    # plot for train data and predict line
    plt.plot(batch_y.data.numpy(), 'ro', label='Original Data')
    plt.plot(predict_y, label='Fitting Line')
    plt.legend()
    plt.show()