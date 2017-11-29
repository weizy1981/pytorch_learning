from sklearn.datasets import load_iris
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch

epochs = 150
batch_size = 150
seed = 7
lr = 0.01
momentum = 0.9

class LogisticLinear(nn.Module):

    # define the model for iris flower dataset
    def __init__(self):
        super(LogisticLinear, self).__init__()
        self.linear = nn.Sequential()
        self.linear.add_module('input', nn.Linear(4, 4))
        self.linear.add_module('hidden1', nn.Linear(4, 6))
        self.linear.add_module('hidden2', nn.Linear(6, 3))
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.linear(x)
        out = self.softmax(out)
        return out

def createTensorDataset(x, y):
    # convert data to Tensors
    x = torch.FloatTensor(x)

    y = np.reshape(y, (y.shape[0], 1))
    # one hot encodering
    enc = OneHotEncoder(sparse=False, n_values=3)
    y = enc.fit_transform(y).astype('float32')
    y = torch.from_numpy(y)

    dataset = TensorDataset(data_tensor=x, target_tensor=y)
    return dataset

if __name__ == '__main__':

    # load data for iris flower dataset
    x, y = load_iris(return_X_y=True)
    labels = y

    # set random seed
    np.random.seed(seed=seed)

    # create model
    model = LogisticLinear()

    # define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)

    dataset = createTensorDataset(x, y)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    # training the model
    for epoch in range(epochs):
        for (data, label) in dataloader:
            # convert to variable
            data = Variable(data)
            label = Variable(label)
            # forward pass
            output = model(data)

            # calculate the loss
            loss = criterion(output, label)
            loss_data = loss.data[0]

            # reset gradients
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            optimizer.step()

            # print loss
            print('{}/{} loss: {:.6f}'.format(epoch, epochs, loss_data))

    model.eval()
    x = torch.FloatTensor(x)
    x = Variable(x)
    outputs = model(x)
    outputs = outputs.data.numpy()
    testacc = 0

    for output, label in zip(outputs, labels):
        predict = np.argmax(output)
        correct_num = (predict == label).sum()
        testacc += correct_num.data[0]

    print("accuary: {:.2f}%".format((testacc / len(labels)) * 100))