import numpy as np
import pandas as pd
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from matplotlib import pyplot as plt 

unlabeled_dir_path = '../../data_hw7'

class Unlabeled(Dataset):
    def __init__(self, path, mode= 'train'):
        super().__init__()
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

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, index):
        image = Image.open(self.image_name[index])
        image = self.toTensor(image)
        return image

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
            img = toPIL(images[index])
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
    images = get_test_image(unlabeled_dir_path)
    plot_images(images, 'test.jpg')

if __name__ == '__main__':
    #test_unlabeled()
    test2()