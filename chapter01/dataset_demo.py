from torch.utils.data import Dataset
from pandas import read_csv

class myDataset(Dataset):

    def __init__(self, csv_file):
        self.csv_data = read_csv(csv_file)
        return

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, item):
        data = self.csv_data[item]
        return data

    def preprocess(self):
        self.target = self.csv_data['y']
        self.data = self.csv_data.drop('y')