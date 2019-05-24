import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class DehazeTrain(Dataset):
    def __init__(self, path, mode= 'train', scene= 'indoor', resize= (512, 512)):
        super().__init__()
        gt_dir   = os.path.join(path, 'Training_GT')
        hazy_dir = os.path.join(path, 'Training_Hazy')
        
        scene_types = ['Indoor', 'Outdoor']
        scene = scene.lower()
        if scene == 'indoor':
            scene_types = ['Indoor']
        elif scene == 'outdoor':
            scene_types = ['Outdoor']

        image_gt_names   = []
        image_hazy_names = []
        for scene_type in scene_types:
            gt_scene_dir = os.path.join(gt_dir, scene_type)
            hazy_scene_dir = os.path.join(hazy_dir, scene_type)

            gt_names = [os.path.join(gt_scene_dir, name) for name in os.listdir(gt_scene_dir)]
            hazy_names = [os.path.join(hazy_scene_dir, name) for name in os.listdir(hazy_scene_dir)]
            gt_names.sort()
            hazy_names.sort()

            valid_size = int(0.2 * len(gt_names))
            if mode == 'train':
                gt_names = gt_names[:-valid_size]
                hazy_names = hazy_names[:-valid_size]
            elif mode == 'valid':
                gt_names = gt_names[-valid_size:]
                hazy_names = hazy_names[-valid_size:]
            
            image_gt_names += gt_names
            image_hazy_names += hazy_names

        image_gt_names.sort()
        image_hazy_names.sort()
        self.gt_names = image_gt_names
        self.hazy_names = image_hazy_names

        self.toTensor = transforms.ToTensor()
        self.resize = resize

    def __len__(self):
        return len(self.gt_names)

    def __getitem__(self, index):
        hazy_img = Image.open(self.hazy_names[index])
        gt_img = Image.open(self.gt_names[index])
        hazy_img, gt_img = self.toTensor(hazy_img), self.toTensor(gt_img)
        
        w_origin, h_origin = gt_img.size(1), gt_img.size(2)
        w, h = self.resize
        w_start = random.randint(0, w_origin - w)
        h_start = random.randint(0, h_origin - h)
        hazy_img = hazy_img[:, w_start : w_start + w, h_start : h_start + h]
        gt_img   = gt_img[:, w_start : w_start + w, h_start : h_start + h]
        return hazy_img, gt_img

class DehazeTest(Dataset):
    def __init__(self, path):
        super().__init__()
        test_dir = os.path.join(path, 'Testing_Images')
        image_names = [os.path.join(test_dir, name) for name in os.listdir(test_dir)]
        image_names.sort()
        self.image_names = image_names
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = Image.open(self.image_names[index])
        image = self.toTensor(image)
        return image

def test():
    dataset_path = '../../DehazeDataset'
    dataset = DehazeTrain(dataset_path, mode= 'all', scene= 'all', resize= (512, 512))
    # print('Size : {}'.format(len(dataset)))
    # hazy_img, gt_img = dataset[5]
    # toPIL = transforms.ToPILImage()
    # hazy_img = toPIL(hazy_img)
    # gt_img  = toPIL(gt_img)
    # hazy_img.show()
    # gt_img.show()

    dataloader = DataLoader(dataset, batch_size= 8)
    for i, data in enumerate(dataloader):
        if i == 5:
            break
        hazy, gt = data
        print(hazy.size())
        print(gt.size())

def test2():
    dataset_path = '../../DehazeDataset'
    dataset = DehazeTest(dataset_path)
    for i in range(len(dataset)):
        img = dataset[i]
        print(img.size())

if __name__ == '__main__':
    test2()