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
        img = Image.open(self.imgs_path[index])
        img_tensor = self.toTensor(img)
        return img_tensor

    def __getitem__(self, index):
        img_tensor = self.get_img_tensor(index)
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

    def get_category_name(self, index):
        return self.category[index]

def test():
    toTensor = transforms.ToTensor()
    toImage = transforms.ToPILImage()
    dataset = MyDataset('../../data_hw5')
    img_norm, _ = dataset[2]
    img_tensor = dataset.get_img_tensor(2)
    img = dataset.toImage(img_norm)
    img = toTensor(img)
    print(torch.sum(img - img_tensor))
    img_new = toTensor(toImage(img))
    print(torch.sum(img_new - img_tensor))

def test2():
    toTensor = transforms.ToTensor()
    toImage = transforms.ToPILImage()
    a = toTensor(Image.open('002.png'))
    b = toTensor(toImage(toTensor(Image.open('002.png'))))
    print(torch.sum((a-b).abs()))

if __name__ == '__main__':
    test()