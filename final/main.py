import sys
sys.path.append('model/')
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import *
from manager import Manager
from basic import Basic, Unet

model = Unet()

parser = argparse.ArgumentParser()
parser.add_argument('mode', help= 'Task: train/predict', choices=['train', 'predict'])
parser.add_argument('-dataset', help= 'Path to dataset', default= '../../DehazeDataset')
parser.add_argument('-bs', help= 'batch size', type= int, default= 8)
parser.add_argument('-lr', help= 'learnig rate', type= float, default= 1e-4)
parser.add_argument('-resize', help= 'Image shape when training', type= int, default= 512)
parser.add_argument('-epoch', help= 'Epoch number', type= int, default= 10000)
parser.add_argument('-check', help= 'Epoch interval for checking performance', type= int, default= 50)
parser.add_argument('-save', help= 'Path to save model')
parser.add_argument('-load', help= 'Path to load model')
parser.add_argument('-output', help= 'Path to output folder')
parser.add_argument('-info', help= 'Infomation about the experiment', default= 'Empty info')
parser.add_argument('-record', help= 'Path to record file')
args = parser.parse_args()

def main():
    if args.mode == 'train':
        print('Training ...')
    
        train_set = DehazeTrain(args.dataset, 'train', 'all', resize= args.resize)
        valid_set = DehazeTrain(args.dataset, 'valid', 'all', resize= args.resize)
        train_data = DataLoader(dataset= train_set, batch_size= args.bs, shuffle= True)
        valid_data = DataLoader(dataset= valid_set, batch_size= args.bs)

        manager = Manager(model, args)
        manager.train(train_data, valid_data)

    else:
        print('Predicting ...')
        test_set = DehazeTest(args.dataset)

        manager = Manager(model, args)
        manager.predict(test_set)

if __name__ == '__main__':
    main()
