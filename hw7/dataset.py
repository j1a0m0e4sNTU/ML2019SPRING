import numpy as np
import pandas as pd
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

unlabeled_dir_path = '../../data_hw7'

class Unlabeled(Dataset):
    def __init__(self, path):
        super().__init__()
        self.test_case = pd.read_csv(os.path.join(path, 'test_case.csv'))
        images_dir = os.path.join(path, 'images')
        image_name = [os.path.join(images_dir, name) for name in os.listdir(images_dir)]
        image_name.sort()
        self.image_name = image_name
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, index):
        image = Image.open(self.image_name[index])
        image = self.toTensor(image)
        #image -= 0.5
        return image

def test_unlabeled():
    data = Unlabeled(unlabeled_dir_path)
    dataloader = DataLoader(data, batch_size=8)
    for i, data in enumerate(dataloader):
        if i == 10:
            break
        print(data.size())

if __name__ == '__main__':
    test_unlabeled()