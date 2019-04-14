import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

class MyDataset(Dataset):
    def __init__(self, path):
        super().__init__()

        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean= self.mean, std= self.std)

        imgs_dir = os.path.join(path, 'images')
        self.imgs_path = [os.path.join(imgs_dir, name) for name in os.listdir(imgs_dir) if name.endswith('.png')]
        self.imgs_path.sort()

        category_csv = pd.read_csv(os.path.join(path, 'categories.csv'))
        self.category = np.array(category_csv['CategoryName'])

        label_csv = pd.read_csv('label.csv')
        self.label = torch.LongTensor(label_csv['label'])

    def __len__(self):
        return len(self.imgs_path)

    def get_img_tensor(self, index):
        img = plt.imread(self.imgs_path[index])
        img_tensor = self.toTensor(img)
        return img_tensor

    def __getitem__(self, index):
        img_tensor = self.get_img_tensor(index)
        img_normalize = self.normalize(img_tensor)
        label = torch.LongTensor([self.label[index]])
        return img_normalize, label
    
    def toNumpy(self, tensor, normalized= True):
        tensor = tensor.squeeze()
        if normalized:
            for i in range(3):
                tensor[i] = (tensor[i] * self.std[i]) + self.mean[i]
        
        array = tensor.permute(1, 2, 0).detach().numpy()
        return array

    def get_category_name(self, index):
        return self.category[index]

def test():
    dataset = MyDataset('../../data_hw5')
    img_tensor = dataset[2]
    img = dataset.toNumpy(img_tensor, True)
    plt.imshow(img)
    plt.show()

def test2():
    dataset = MyDataset('../../data_hw5')
    for i in range(len(dataset)):
        img, label = dataset[i]
        category = dataset.get_category_name(label.item())
        print('{}, {}'.format(i, category))

if __name__ == '__main__':
    test2()