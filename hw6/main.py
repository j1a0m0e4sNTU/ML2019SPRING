import argparse
import torch
from model import get_rnn_model
from util import *
from manager import Manager

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices= ['train', 'predict'])
parser.add_argument('config', help= 'Symbol of model configuration')
parser.add_argument('-train_x', help= 'Path to train_x.csv', default= '../../data_hw6/train_x.csv')
parser.add_argument('-train_y', help= 'Path to train_y.csv', default= '../../data_hw6/train_y.csv')
parser.add_argument('-test_x', help= 'Path to test_x.csv', default= '../../data_hw6/test_x.csv')
parser.add_argument('-dict', help= 'Path to dictionary', default= '../../data_hw6/dict.txt.big')
parser.add_argument('-word_model', help= 'Path to Word model', default= '../../data_hw6/word2vec.model')
parser.add_argument('-lr', help= 'learnig rate', type= float, default= 1e-4)
parser.add_argument('-epoch', help= 'Epoch number', type= int, default= 10)
parser.add_argument('-save', help= 'Path to save model')
parser.add_argument('-load', help= 'Path to load model')
parser.add_argument('-record', help= 'Path to file for recording result')
parser.add_argument('-predcit', help= 'Path to prediction file')
args = parser.parse_args()

if __name__ == '__main__':
    model = get_rnn_model(args.config)
    if args.mode == 'train':
        print('= Training =')
        train_data = WordsData(mode= 'train', x_path= args.train_x, y_path= args.train_y,
                                model_path= args.word_model, dict_path= args.dict)
        valid_data = WordsData(mode= 'valid', x_path= args.train_x, y_path= args.train_y,
                                model_path= args.word_model, dict_path= args.dict)
        
        model_manager = Manager(model, args)
        model_manager.train(train_data, valid_data)

    elif args.mode == 'predict':
        print('= Predicting =')
        test_data = WordsData(mode= 'test', x_path= args.test_x, y_path= None,
                                model_path= args.word_model, dict_path= args.dict)
        
        model_manager = Manager(model, args)
        model_manager.predict(test_data, args.predict)