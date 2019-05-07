import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import get_rnn_model
from util import *
from manager import Manager
torch.manual_seed(1004)

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices= ['train', 'predict', 'ensemble'])
parser.add_argument('config', help= 'Symbol of model configuration', default= 'simple')
parser.add_argument('-train_x', help= 'Path to train_x.csv', default= '../../data_hw6/train_x.csv')
parser.add_argument('-train_y', help= 'Path to train_y.csv', default= '../../data_hw6/train_y.csv')
parser.add_argument('-test_x', help= 'Path to test_x.csv', default= '../../data_hw6/test_x.csv')
parser.add_argument('-dict', help= 'Path to dictionary', default= '../../data_hw6/dict.txt.big')
parser.add_argument('-word_model', help= 'Path to Word model', default= '../../data_hw6/word2vec.model')
parser.add_argument('-lr', help= 'learnig rate', type= float, default= 1e-3)
parser.add_argument('-epoch', help= 'Epoch number', type= int, default= 7)
parser.add_argument('-batch_size', type= int, default= 32)
parser.add_argument('-seq_len', type= int, help= 'Sequence length for each data', default= 30)
parser.add_argument('-save', help= 'Path to save model')
parser.add_argument('-load', help= 'Path to load model')
parser.add_argument('-record', help= 'Path to file for recording result')
parser.add_argument('-predict', help= 'Path to prediction file')
args = parser.parse_args()

if __name__ == '__main__':
    model = get_rnn_model(args.config, args.batch_size)
    if args.mode == 'train':
        print('= Training =')
        train_words = WordsData(mode= 'train', x_path= args.train_x, y_path= args.train_y,
                                model_path= args.word_model, dict_path= args.dict, seq_len= args.seq_len)
        valid_words = WordsData(mode= 'valid', x_path= args.train_x, y_path= args.train_y,
                                model_path= args.word_model, dict_path= args.dict, seq_len= args.seq_len)
        train_data = DataLoader(train_words, batch_size= args.batch_size, shuffle= True)
        valid_data = DataLoader(valid_words, batch_size= args.batch_size, shuffle= False)

        model_manager = Manager(model, args)
        model_manager.train(train_data, valid_data)

    elif args.mode == 'predict':
        print('= Predicting =')
        test_words = WordsData(mode= 'test', x_path= args.test_x, y_path= None,
                                model_path= args.word_model, dict_path= args.dict, seq_len= args.seq_len)
        test_data = DataLoader(test_words, batch_size= args.batch_size, shuffle= False)

        model_manager = Manager(model, args)
        model_manager.predict(test_data, args.predict)
    
    elif args.mode == 'ensemble':
        test_num = 20000
        seq_len_list = [20, 40, 60, 80, 100]
        weight_dir = '../../weights/'
        model_weight_pair = {
            'simple': '0507_2.pkl',
            'A': '0507_11.pkl',
            'B': '0507_12.pkl',
            'C': '0507_13.pkl'
        }

        scores = np.zeros((test_num,))
        threshold = len(seq_len_list) * len(model_weight_pair) / 2
        for m_name, w_name in model_weight_pair.items():
            model = get_rnn_model(m_name, args.batch_size)
            args.load = os.path.join(weight_dir, w_name)
            model_manager = Manager(model, args)
            for seq_len in seq_len_list:
                test_words = WordsData(mode= 'test', x_path= args.test_x, y_path= None,
                                model_path= args.word_model, dict_path= args.dict, seq_len= seq_len)
                test_data = DataLoader(test_words, batch_size= args.batch_size, shuffle= False)
                scores += model_manager.get_all_predictions(test_data)

        file = open(args.predict, 'w')
        file.write('id,label\n')
        for i, score in enumerate(scores):
            pred = 1 if score > threshold else 0
            file.write('{},{}\n'.format(i, pred))
