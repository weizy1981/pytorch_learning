from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from pandas import read_csv, get_dummies, concat, DataFrame
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

filename = 'pollution_original.csv'
time_step = 24
batch_size = 128
epochs = 1
lr = 0.01
feature = 11

# combine the datetime
def prase(x):
    return datetime.strptime(x, '%Y %m %d %H')

def get_dataset():
    dataset = read_csv(filename, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=prase)
    dataset.drop('No', axis=1, inplace=True)
    # set the column name
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    # one hot encode
    df = get_dummies(dataset['wnd_dir'])
    dataset = concat((dataset, df), axis=1)
    dataset.drop('wnd_dir', axis=1, inplace=True)
    dataset.dropna(inplace=True)

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    return dataset

def make_dataset(data, n_input=1, out_index=0, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []
    # input (t-n, ... t-1)
    for i in range(n_input, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # output (t)
    cols.append(df[df.columns[out_index]])
    names += ['result']
    # combine the inputs
    result = concat(cols, axis=1)
    result.columns = names
    # del miss values cols
    if dropnan:
        result.dropna(inplace=True)
    return result

def createTensorDataset(data):
    values = data.values
    x = values[:, : values.shape[1] - 1]
    y = values[:, values.shape[1] - 1]

    # convert data to Tensors
    x = torch.FloatTensor(x)
    y = np.reshape(y, (y.shape[0], 1)).astype('float32')
    y = torch.from_numpy(y)

    # create Tensor Dataset
    dataset = TensorDataset(data_tensor=x, target_tensor=y)
    return dataset

class LSTMPollution(nn.Module):

    def __init__(self):
        super(LSTMPollution, self).__init__()
        # define the model
        self.lstms1 = nn.LSTM(input_size=(feature * time_step), hidden_size=33, num_layers=5)
        self.linear1 = nn.Linear(in_features=33, out_features=1)

    def forward(self, x):
        out, _ = self.lstms1(x)
        # b, s, h = out.size()
        out = out[:, -1]
        out = self.linear1(out)
        return out


if __name__ == '__main__':
    np.random.seed(7)

    # read data from csv
    data = get_dataset()

    # create dataset for lstm
    dataset = make_dataset(data, n_input=time_step)
    dataset = dataset[0:365]

    # create Tensor Dataset
    dataset = createTensorDataset(dataset)

    model = LSTMPollution()

    # define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

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
            print('{}/{} loss: {:.6f}'.format(epoch + 1, epochs, loss_data))

    model.eval()
    x = dataset.data_tensor
    x = Variable(x)
    y = dataset.target_tensor.numpy()
    outputs = model(x)
    outputs = outputs.data.numpy()
