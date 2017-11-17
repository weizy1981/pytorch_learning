from torch.utils.data import Dataset
from pandas import read_csv

csv_file = 'filepath'
class MyDataset(Dataset):

    def __init__(self):
        self.csv_data = read_csv(csv_file)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, item):
        data = self.csv_data[item]
        return data

    def preprocess(self):
        self.target = self.csv_data['y']
        self.data = self.csv_data.drop('y')

from torch.utils.data import DataLoader

myDataset = MyDataset()
dataloader = DataLoader(myDataset, batch_size=32, shuffle=True)