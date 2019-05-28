import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import  torchvision.transforms as transforms

class TrainDataset(Dataset):
    def __init__(self, path, mode, transform= None):
        self.transform = transform
        file = pd.read_csv(path)
        
        label = []
        feature = []
        count = 0
        for i in range(len(file)):
            l = int(file['label'][i])
            f = str(file['feature'][i])
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
        feature = torch.tensor(feature_f)
        feature = feature.view(1, 48, 48)
        if self.transform:
            feature = self.transform(feature)
    
        return label, feature
    
    def __len__(self):
        return len(self.label)

class TestDataset(Dataset):
    def __init__(self, path, mode= None, transform= None):
        self.transform = transform
        file = pd.read_csv(path)
        data_num = len(file)
   
        self.feature = []
        for i in file['feature']:
            self.feature.append(i)

    def __getitem__(self, index):
        feature_str = self.feature[index].split(' ')
        feature_f = [float(i) for i in feature_str]
        feature = torch.tensor(feature_f)
        feature = feature.view(1, 48, 48)
        if self.transform:
            feature = self.transform(feature)

        return feature
    
    def __len__(self):
        return len(self.feature)


def test_traindata():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees= 10, translate= (0.1, 0.1), scale= (0.9, 1.1)),
        transforms.ToTensor()
    ])

    faces = TrainDataset('../../data_hw8/train.csv', mode= 'train', transform= transform)
    data = DataLoader(faces, batch_size= 8)
    print(len(faces))
    for pair in data:
        label, imgs = pair
        print(label)        
        print(imgs[0])
        break

def test_testdata():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.TenCrop(44),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    ])

    faces = TestDataset('../../data_hw8/test.csv', transform = transform)
    data = DataLoader(faces, batch_size= 8)
    for i, feature in enumerate(data):
        print(feature.size())
        print(feature[0, 0, 0])
        break

def test2():
    file = pd.read_csv('../../data_hw8/train.csv')
    count = np.zeros((7))
    for i in file['label']:
        index = int(i)
        count[index] += 1

    print(count)
    

if __name__ == '__main__':
    test_traindata()
    