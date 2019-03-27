import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    def __init__(self, path, mode= None, normalize= True):
        self.normalize = normalize

        file = pd.read_csv(path)
        
        label = []
        feature = []
        count = 0
        for i in range(len(file)):
            l = int(file['label'][i])
            f = str(file['feature'][i])
            # if l == 6:
            #     continue
            label.append(l)
            feature.append(f)
            count += 1

        train_num = int(0.8 * count)
        if mode == 'train':
            self.label = label[:train_num]
            self.feature = feature[:train_num]
        elif mode == 'valid':
            self.label = label[train_num:]
            self.feature = feature[train_num:]


    def __getitem__(self, index):
        label = torch.tensor(self.label[index], dtype= torch.long)
        feature_str = self.feature[index].split(' ')
        feature_f = [float(i) for i in feature_str]
        feature = torch.tensor(feature_f, dtype= torch.float)
        feature = feature.view(1, 48, 48)
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
        self.feature = []
        for i in file['feature']:
            self.feature.append(i)

    def __getitem__(self, index):
        feature_str = self.feature[index].split(' ')
        feature_f = [float(i) for i in feature_str]
        feature = torch.tensor(feature_f, dtype= torch.float)
        feature = feature.view(1, 48, 48)
        if self.normalize:
            feature = (feature - 128) / 128
        return feature
    
    def __len__(self):
        return len(self.feature)


def test():
    faces = TrainDataset('../../data_hw3/train.csv', mode='train')
    data = DataLoader(faces, batch_size= 8)
    print(len(faces))
    for pair in data:
        label, imgs = pair
        print(label)        
        print(imgs)
        break

def test2():
    file = pd.read_csv('../../data_hw3/train.csv')
    count = np.zeros((7))
    for i in file['label']:
        index = int(i)
        count[index] += 1

    print(count)
    

if __name__ == '__main__':
    test2()
