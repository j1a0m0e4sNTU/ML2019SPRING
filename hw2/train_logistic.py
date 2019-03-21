import argparse
import numpy as np
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('-csv', default= 'data/train.csv', help= 'path to train.csv')
parser.add_argument('-x', default= 'data/X_train', help= 'Path to X_train')
parser.add_argument('-y', default= 'data/Y_train', help= 'Path to Y_train')
parser.add_argument('-steps', type= int, default= 500, help= 'Training step number')
parser.add_argument('-lr', type= float, default= 1e-1, help= 'Learning rate')
parser.add_argument('-check', type= int, default= 100, help= 'epoch number to check performance')
parser.add_argument('-th', type= float, default= 0.5, help= 'Threshold to determine 0/1')
parser.add_argument('-regularize', type= float, default= 0, help= 'Regularization weight')
parser.add_argument('-validate', type= int, default= 1, help= 'Validate or not')
parser.add_argument('-save', default= None, help= 'Weights name')
args = parser.parse_args()

def main():
    total_x = get_total_feature(args.csv, args.x)
    mean = np.mean(total_x, 0)
    std = np.std(total_x, 0)
    total_x = normalize_feature(total_x, mean, std)
    total_x = discretalize_all(total_x)
    total_x = add_constant_column(total_x)
    total_y = get_raw_data(args.y)
    
    if args.validate:
        train_acc, valid_acc = 0, 0

        for fold in range(5):
            print('=======',fold, '======')
            train_x, train_y, valid_x, valid_y = get_train_valid_data(total_x, total_y, fold)
            t_acc, v_acc = train_logistic(train_x, train_y, valid_x, valid_y)
            train_acc += t_acc
            valid_acc += v_acc
        print(' ---------------------------------')
        print('Training Acc: ', train_acc/5, 'Validation Acc: ', valid_acc/5)

    else:
        train_x, train_y, valid_x, valid_y = total_x, total_y, None, None
        train(total_x, total_y, None, None)

def train_logistic(train_x, train_y, valid_x, valid_y):
    dim = train_x.shape[1]
    weight = np.zeros((dim), dtype= np.float)
    learning_rate = args.lr
    train_x_T = train_x.T
    grad_prev = 0
    train_acc, valid_acc = 0, 0
    
    for step in range(args.steps):
       
        gradient_weight = (-1) * (train_y - sigmoid(train_x @ weight))
        gradient = train_x_T @ gradient_weight + args.regularize * weight
        grad_prev += gradient ** 2
        ada = np.sqrt(grad_prev) + 1e-5
        weight -= learning_rate * (gradient / ada) 
        
        if (step + 1) % args.check == 0:
            train_pred = sigmoid(train_x @ weight)
            train_acc = compute_acc(train_pred, train_y)
            print('Step', step + 1, '| Training Acc:', train_acc, end=' ')
            if args.validate:
                valid_pred = sigmoid(valid_x @ weight)
                valid_acc = compute_acc(valid_pred, valid_y)
                print('| Validation acc:', valid_acc, end='')
            print()
    
    if args.save:
        np.save(args.save, weight)

    return train_acc, valid_acc


def compute_acc(pred, target):
    pred[pred > args.th] = 1
    pred[pred <=args.th] = 0

    total = pred.shape[0]
    correct = np.sum(pred == target)
    return correct / total


if __name__ == '__main__':
    main()