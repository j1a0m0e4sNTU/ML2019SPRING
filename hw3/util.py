import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    def __init__(self, path, mode= None, normalize= True):
        file = pd.read_csv(path)
        data_num = len(file)
        
        self.normalize = normalize
        self.label = file['label']
        self.feature = file['feature']

        train_num = int(0.8 * data_num)
        if mode == 'train':
            self.label = self.label[:train_num]
            self.feature = self.feature[:train_num]
        elif mode == 'valid':
            self.label = self.label[train_num:]
            self.feature = self.feature[train_num:]

    def __getitem__(self, index):
        label = torch.tensor(self.label[index], dtype= torch.long)
        feature_str = self.feature[index].split(' ')
        feature_f = [float(i) for i in feature_str]
        feature = torch.tensor(feature_f, dtype= torch.float)
        feature = feature.view(48, 48)
        if self.normalize:
            feature = (feature - 128) / 128
        return label, feature
    
    def __len__(self):
        return len(self.label)

class TestDataset(Dataset):
    def __init__(self, path, mode= None, normalize= True):
        file = pd.read_csv(path)
        data_num = len(file)
        
        self.normalize = normalize
        self.feature = file['feature']

    def __getitem__(self, index):
        feature_str = self.feature[index].split(' ')
        feature_f = [float(i) for i in feature_str]
        feature = torch.tensor(feature_f, dtype= torch.float)
        feature = feature.view(48, 48)
        if self.normalize:
            feature = (feature - 128) / 128
        return feature
    
    def __len__(self):
        return len(self.feature)


def test():
    faces = TrainDataset('../data/hw3/train.csv')
    data = DataLoader(faces, batch_size= 8)
    for pair in data:
        label, imgs = pair
        print(label.shape)
        break
    print(len(data))

if __name__ == '__main__':
    test()