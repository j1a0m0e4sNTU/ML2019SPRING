import numpy as np
import pandas as pd
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms 
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

unlabeled_dir_path = '../../data_hw7'
test_case_path = '../../data_hw7/test_case.csv'

class Unlabeled(Dataset):
    def __init__(self, path, mode= 'train'):
        super().__init__()
        self.path = path
        self.toTensor = transforms.ToTensor()
        self.test_case = pd.read_csv(os.path.join(path, 'test_case.csv'))
        images_dir = os.path.join(path, 'images')
        image_name = [os.path.join(images_dir, name) for name in os.listdir(images_dir)]
        image_name.sort()
        cut_size = int(len(image_name) * 0.8)
        if mode == 'train':
            image_name = image_name[:cut_size]
        elif mode == 'valid':
            image_name = image_name[cut_size:]
        self.image_name = image_name
        test_case_csv = pd.read_csv(os.path.join(path, 'test_case.csv'))
        test_case = np.zeros((len(test_case_csv), 2)).astype(np.int)
        image1_name = [os.path.join(images_dir, '{:0>6d}.jpg'.format(index)) for index in test_case_csv['image1_name']]
        image2_name = [os.path.join(images_dir, '{:0>6d}.jpg'.format(index)) for index in test_case_csv['image2_name']]
        self.test_case = np.array([image1_name, image2_name])

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, index):
        image = Image.open(self.image_name[index])
        image = self.toTensor(image)
        return image

    def get_test_case(self, index):
        img1 = Image.open(self.test_case[0, index])
        img2 = Image.open(self.test_case[1, index])
        img1, img2 = self.toTensor(img1), self.toTensor(img2)
        return img1, img2

def get_test_image(path):
    images_dir = os.path.join(path, 'images')
    image_name = [os.path.join(images_dir, name) for name in os.listdir(images_dir)]
    image_name.sort()
    image_name = image_name[-32:]
    images = torch.zeros(32, 3, 32, 32)
    toTensor = transforms.ToTensor()
    for i in range(len(image_name)):
        image = Image.open(image_name[i])
        images[i] = toTensor(image)
    return images

def plot_images(images, name):
    toPIL = transforms.ToPILImage()
    for i in range(4):
        for j in range(8):
            index = i * 8 + j
            img = toPIL(images[index].cpu().detach())
            img = np.array(img)
            plt.subplot(4, 8, index + 1)
            plt.imshow(img)
            plt.xticks([], [])
            plt.yticks([], [])
            
    plt.savefig(name)
    plt.close()
            
def test_unlabeled():
    data = Unlabeled(unlabeled_dir_path)
    dataloader = DataLoader(data, batch_size=8)
    for i, data in enumerate(dataloader):
        if i == 10:
            break
        print(data.size())

def test():
    train_data = Unlabeled(unlabeled_dir_path, 'train')
    print(len(train_data))
    valid_data = Unlabeled(unlabeled_dir_path, 'valid')
    print(len(valid_data))

def test2():
    train_data = Unlabeled(unlabeled_dir_path, 'train')
    img1, img2 = train_data.get_test_case(0)
    print(img1)
    print(img1.size())

if __name__ == '__main__':
    #test_unlabeled()
    test2()
