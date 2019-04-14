import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

class MyDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean= [0.485, 0.456, 0.406], 
                                                                std= [0.229, 0.224, 0.225])])

        imgs_dir = os.path.join(path, 'images')
        self.imgs_path = [os.path.join(imgs_dir, name) for name in os.listdir(imgs_dir) if name.endswith('.png')]
        self.imgs_path.sort()

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = plt.imread(self.imgs_path[index])
        img = self.transform(img)
        return img

    def get_original(self, index):
        img_origin = plt.imread(self.imgs_path[index])
        img_transform = self.transform(img_origin)
        return img_origin, img_transform

if __name__ == '__main__':
    dataset = MyDataset('../../data_hw5')