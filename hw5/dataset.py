import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, path):
        super().__init__()

        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean= self.mean, std= self.std)
        self.toPIL = transforms.ToPILImage()

        self.imgs_path = [os.path.join(path, name) for name in os.listdir(path) if name.endswith('.png')]
        self.imgs_path.sort()

        label_csv = pd.read_csv('label.csv')
        self.label = torch.LongTensor(label_csv['label'])

    def __len__(self):
        return len(self.imgs_path)


    def get_img_data(self, index):
        img = Image.open(self.imgs_path[index])
        img_array = np.array(img)
        img_data = torch.from_numpy(img_array).type(torch.float)
        return img_data

    def __getitem__(self, index):
        img = Image.open(self.imgs_path[index])
        img_tensor = self.toTensor(img)
        img_normalize = self.normalize(img_tensor)
        label = torch.LongTensor([self.label[index]])
        return img_normalize, label
    
    def toImage(self, tensor, normalized= True):
        tensor = tensor.squeeze()
        image = torch.zeros_like(tensor)
        if normalized:
            for i in range(3):
                image[i] = (tensor[i] * self.std[i]) + self.mean[i]
        
        image = self.toPIL(image)
        return image

    def transform(self, image):
        # Input: numpy array or PIL image
        img_tensor = self.toTensor(image)
        img_normal = self.normalize(img_tensor)
        return img_normal

class Category():
    def __init__(self, path= '../../data_hw5/categories.csv'):
        category_csv = pd.read_csv(path)
        self.category = np.array(category_csv['CategoryName'])

    def __getitem__(self, index):
        return self.category[index]

def test():
    dataset = MyDataset('../../data_hw5/images')
    for i in range(10):
        tensor, label = dataset[i]
        print(tensor.size(), label)

def test3():
    csv_pred = pd.read_csv('label.csv')
    label_pred = np.array(csv_pred['label'])
    csv_gt = pd.read_csv('../../data_hw5/labels.csv')
    label_gt = np.array(csv_gt['TrueLabel'])

    dataset = MyDataset('../../data_hw5/images')
    count = 0
    for i in range(200):
        if label_pred[i] != label_gt[i]:
            count += 1
            cate_pred = dataset.get_category_name(label_pred[i])
            cate_gt = dataset.get_category_name(label_gt[i])
            print('{:0>3d}.png | ground truth: {} prediction: {}'.format(i, cate_gt, cate_pred))

    print('Total difference count: {}'.format(count))

def test4():
    dataset = MyDataset('../../data_hw5/images')
    img = dataset.get_img_data(0)
    print(img)

if __name__ == '__main__':
    test4()