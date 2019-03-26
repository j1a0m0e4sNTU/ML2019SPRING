import argparse
import torch
from torch.utils.data import DataLoader
import util
import manager
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('mode', help= 'Task: train/predict', choices=['train', 'predict'])
parser.add_argument('-dataset', help= 'Path to dataset', default= '../data/hw3/train.csv')
parser.add_argument('-normal', help= 'Normalize data or not', type= bool, default= True)
parser.add_argument('-bs', hekp= 'batch size', type= int, default= 64)
parser.add_argument('-lr', help= 'learnig rate', type= float, default= 1e-3)
parser.add_argument('-epoch', help= 'Epoch number', type= int, default= 20)
parser.add_argument('-save', help= 'Path to save model')
parser.add_argument('-load', help= 'Path to load model')
parser.add_argument('-predict', help= 'Path to prediction file')
args = parser.parse_args()

def main():
    model = Model_basic()
    if args.mode == 'train':
        print('Training ...')
        train_set = TrainDataset(args.dataset, mode= 'train', normalize= args.normal)
        valid_set = TrainDataset(args.dataset, mode= 'valid', normalize= args.normal)
        train_data = DataLoader(dataset= train_set, batch_size= args.bs, shuffle= True)
        valid_data = DataLoader(dataset= valid_set, batch_size= args.bs)

        manager = Manager(model, args)
        manager.train(train_data, valid_data)

    else:
        print('Predicting ...')
        test_set = TestDataset(args.dataset, normalize= args.normal)
        test_data = DataLoader(dataset= test_set, batch_size= 1)

        manager = Manager(model, args)
        manager.predict(args.predict)