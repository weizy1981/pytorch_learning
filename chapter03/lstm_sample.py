from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from pandas import read_csv, get_dummies, concat, DataFrame
from datetime import datetime

filename = 'pollution_original.csv'
time_step = 1
batch_size = 5
epochs = 100
lr = 0.01

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
    y = np.reshape(y, (y.shape[0], 1))
    y = torch.from_numpy(y)

    # create Tensor Dataset
    dataset = TensorDataset(data_tensor=x, target_tensor=y)
    return dataset

if __name__ == '__main__':
    # read data from csv
    data = get_dataset()

    # create dataset for lstm
    dataset = make_dataset(data, n_input=time_step)

    # create Tensor Dataset
    dataset = createTensorDataset(dataset)
