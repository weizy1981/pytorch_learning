import numpy as np
from torch import nn
import torch
from torch import optim
from torch.autograd import Variable
from matplotlib import pyplot as plt

lr = 0.01
epochs = 100


class LinearRegression(nn.Module):
    def __init__(self):
        # define linear layer
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

if __name__ == '__main__':
    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.186],
                       [9.779], [6.182], [7.59], [2.167], [7.042], [10.791],
                       [5.313], [7.997], [3.1]], dtype=np.float32)

    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827], [3.465],
                        [1.65], [2.904], [1.3]], dtype=np.float32)

    # convert ndarray to variable
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    if torch.cuda.is_available():
        model = LinearRegression().cuda()
    else:
        model = LinearRegression()

    # define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # train Linear Regression
    for epoch in range(epochs):
        if torch.cuda.is_available():
            inputs = Variable(x_train).cuda()
            target = Variable(y_train).cuda()
        else:
            inputs = Variable(x_train)
            target = Variable(y_train)

        # forward
        out = model(inputs)
        loss = criterion(out, target)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch {}/{}, loss: {:.6f}'.format(epoch + 1, epochs,
                                                 loss.data[0]))

    # Predict
    model.eval()
    y_predict = model(Variable(x_train))
    y_predict = y_predict.data.numpy()

    # plot for train data and predict line
    plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original Data')
    plt.plot(x_train.numpy(), y_predict, label='Fitting Line')
    plt.show()