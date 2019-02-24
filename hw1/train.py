import numpy as np
import argparse
from feature_extract import *
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('-step', help='Training steps', type= int, default= 100000)
parser.add_argument('-validation', help='Validation rate', type= float, default= 0.2)
parser.add_argument('-lr', help= 'Learning rate', type= float, default= 1e-2)
parser.add_argument('-save', help= 'Weight name to be saved', default= None)
parser.add_argument('-check', help='Steps number to check performance', type= int, default= 1000)
parser.add_argument('-regularize', help= 'Regularization rate', type= float, default= 1e-4)
args = parser.parse_args()

def train():
    train_data = get_aligned_train_data()
    extractor = basic_extractor()
    train_x, train_y, valid_x, valid_y = get_split_data(train_data, extractor, args.validation)

    train_x_T, valid_x_T = train_x.T, valid_x.T
    weight = np.zeros((extractor.feature_num, ), dtype= np.float) 
    learning_rate = args.lr
    grad_prev = 0

    for i in range(args.step):

        gradient_weight = (-2) * (train_y - train_x @ weight)
        gradient = train_x_T @ gradient_weight
        grad_prev += gradient ** 2
        ada = np.sqrt(grad_prev)
        weight = weight - learning_rate * (gradient / ada)

        if (i+1) % args.check == 0:
            train_pred = train_x @ weight
            train_error = get_mse_error(train_pred, train_y)
            print('After ', i+1, 'steps... | Training error:',train_error, end=' ')
            if args.validation > 0:
                valid_pred = valid_x @ weight
                valid_error = get_mse_error(valid_pred, valid_y)
                print('| Validation error:', valid_error, end=' ')
            print()
        
    if args.save:
        np.save(args.save, weight)

if __name__ == '__main__':
    train()