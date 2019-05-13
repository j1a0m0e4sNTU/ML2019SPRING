import argparse
import torch
from torch.utils.data import DataLoader
from dataset import *
from manager import Manager
from model import AutoEncoder
torch.manual_seed(1004)

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train', 'predict'])
parser.add_argument('-E', help= 'Encoder symbol', default= 'base')
parser.add_argument('-D', help= 'Decoder symbol', default= 'base')
parser.add_argument('-dataset', help= 'Path to dataset', default= '../../data_hw7')
parser.add_argument('-bs', help= 'batch size', type= int, default= 64)
parser.add_argument('-lr', help= 'learnig rate', type= float, default= 1e-4)
parser.add_argument('-epoch', help= 'Epoch number', type= int, default= 30)
parser.add_argument('-save', help= 'Path to save model')
parser.add_argument('-load', help= 'Path to load model')
parser.add_argument('-csv', help= 'Path to prediction')
parser.add_argument('-record', help= 'Path to record file', default='records/test.txt')
args = parser.parse_args()

if __name__ == '__main__':
    if args.mode == 'train':
        print('======= Training ========')
        manager = Manager(args)
        data_train = DataLoader(Unlabeled(args.dataset, 'train'), batch_size= args.bs, shuffle= True)
        data_valid = DataLoader(Unlabeled(args.dataset, 'valid'), batch_size= args.bs, shuffle= False)
        manager.train(data_train, data_valid)    

    elif args.mode == 'predict':
        pass
