import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from util import *
from manager import Manager
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('mode', help= 'Task: train/predict', choices=['train', 'predict'])
parser.add_argument('-dataset', help= 'Path to dataset', default= '../../data_hw3/train.csv')
parser.add_argument('-normal', help= 'Normalize data or not', type= int, default= 1)
parser.add_argument('-bs', help= 'batch size', type= int, default= 64)
parser.add_argument('-lr', help= 'learnig rate', type= float, default= 1e-4)
parser.add_argument('-epoch', help= 'Epoch number', type= int, default= 10)
parser.add_argument('-save', help= 'Path to save model')
parser.add_argument('-load', help= 'Path to load model')
parser.add_argument('-predict', help= 'Path to prediction file')
args = parser.parse_args()

def main():
    model = Model_1()
    if args.mode == 'train':
        print('Training ...')
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])
        valid_transform = transforms.Compose([
            transforms.ToPILImage(),
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
            transforms.ToTensor()
        ])
        test_set = TestDataset(args.dataset, transform= test_transform)
        test_data = DataLoader(dataset= test_set, batch_size= 1)

        manager = Manager(model, args)
        manager.predict(args.predict)

if __name__ == '__main__':
    main()
