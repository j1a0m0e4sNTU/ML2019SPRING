import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from util import *
from manager import Manager
from model import *
from model_vgg import *

model = get_DNN('DNN')

parser = argparse.ArgumentParser()
parser.add_argument('mode', help= 'Task: train/predict', choices=['train', 'predict'])
parser.add_argument('-dataset', help= 'Path to dataset', default= '../../data_hw3/train.csv')
parser.add_argument('-bs', help= 'batch size', type= int, default= 64)
parser.add_argument('-lr', help= 'learnig rate', type= float, default= 1e-4)
parser.add_argument('-epoch', help= 'Epoch number', type= int, default= 10)
parser.add_argument('-save', help= 'Path to save model')
parser.add_argument('-load', help= 'Path to load model')
parser.add_argument('-csv', help= 'Path to prediction file')
parser.add_argument('-record', help= 'Path to record file')
parser.add_argument('-tencrop', help= 'Apply TenCrop to testdata', type= int, default= 0)
args = parser.parse_args()

def main():
    if args.mode == 'train':
        print('Training ...')
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(44),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees= 10, scale= (0.9, 1.1)),
            transforms.ToTensor()
        ])
        valid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(44),
            transforms.ToTensor()
        ])
        train_set = TrainDataset(args.dataset, mode= 'train', transform= train_transform)
        valid_set = TrainDataset(args.dataset, mode= 'valid', transform= valid_transform)
        train_data = DataLoader(dataset= train_set, batch_size= args.bs, shuffle= True)
        valid_data = DataLoader(dataset= valid_set, batch_size= args.bs)

        manager = Manager(model, args)
        manager.train(train_data, valid_data)

    else:
        print('Predicting ...')
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.TenCrop(44),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ])
        test_set = TestDataset(args.dataset, transform= test_transform)
        test_data = DataLoader(dataset= test_set, batch_size= args.bs)

        manager = Manager(model, args)
        manager.predict(test_data)

if __name__ == '__main__':
    main()
