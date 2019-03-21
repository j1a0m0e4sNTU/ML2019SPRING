import argparse
import numpy as np
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('-train_csv', default= '../data/hw2/train.csv', help= 'path to train.csv')
parser.add_argument('-train_x', default= '../data/hw2/X_train', help= 'Path to X_train')
parser.add_argument('-train_y', default= '../data/hw2/Y_train', help= 'Path to Y_train')
parser.add_argument('-test_csv', default= '../data/hw2/test.csv', help= 'path to test.csv')
parser.add_argument('-test_x', default= '../data/hw2/X_test', help= 'Path to X_test')
parser.add_argument('-validate', type= int, default= 1, help= 'Validate or not')
parser.add_argument('-save', default= None, help= 'Path to prediction csv')
args = parser.parse_args()

total_x = get_total_feature(args.train_csv, args.train_x)
mean = np.mean(total_x, 0)
std  = np.std(total_x, 0)
total_x = (total_x - mean) / std
total_y = get_raw_data(args.train_y)

test_x = get_total_feature(args.test_csv, args.test_x)
test_x = (test_x - mean) / std

def main():
    if args.validate:

        train_acc, valid_acc = 0, 0
        for fold in range(5):
            print('=======',fold, '======')
            train_x, train_y, valid_x, valid_y = get_train_valid_data(total_x, total_y, fold)
            w, b = train(train_x, train_y)
            t_acc = get_acc(w, b, train_x, train_y)
            v_acc = get_acc(w, b, valid_x, valid_y)
            print('train acc:', t_acc, 'valid acc:', v_acc)
            train_acc += t_acc
            valid_acc += v_acc
        
        print('---------------------------------')
        print('Training Acc:  ', train_acc / 5)
        print('Validation Acc:', valid_acc / 5)

    else:
        w ,b = train(total_x, total_y)
        prediction = predict(w, b, test_x)
        num = test_x.shape[0]

        file = open(args.save, 'w')
        file.write('id,label\n')
        for i in range(num):
            line = str(i + 1) + ',' + str(int(prediction[i])) + '\n'
            file.write(line)
    
        print('Prediction complete !')

def train(train_x, train_y):
    train_x_1 = train_x[train_y == 1]
    train_x_0 = train_x[train_y == 0]
    mu_1 = np.mean(train_x_1, 0)
    mu_0 = np.mean(train_x_0, 0)
    n_1, n_0 = train_x_1.shape[0], train_x_0.shape[0]

    covariance_M1 = get_covariance_matrix(train_x_1, mu_1)
    covariance_M0 = get_covariance_matrix(train_x_0, mu_0)
    covariance_M = (n_1 / (n_1 + n_0)) * covariance_M1 + (n_0 / (n_1 + n_0)) * covariance_M0
    
    mu_1 = mu_1.reshape(-1, 1)
    mu_0 = mu_0.reshape(-1, 1)
    M_inverse = np.linalg.inv(covariance_M)
    w = (mu_1 - mu_0).T @ M_inverse
    b = (-1/2) * (mu_1.T @ M_inverse @ mu_1) + (1/2) * (mu_0.T @ M_inverse @ mu_0) + np.log(n_1/n_0) 

    return w, b
    
def get_covariance_matrix(x, mu):
    data_num, f_num = x.shape
    matrix = np.zeros((f_num, f_num))

    for i in range(data_num):
        vector = (x[i] - mu).reshape(-1, 1)
        matrix += vector @ (vector.T)
    
    matrix /= data_num
    return matrix

def predict(w, b, x):
    score = sigmoid(x @ w.T + b)
    score = score.reshape(-1,)
    prediction = np.zeros((x.shape[0]))
    prediction[score >= 0.5] = 1
    prediction[score <  0.5] = 0 
    return prediction

def get_acc(w, b, x, y):
    pred = predict(w, b, x)
    same = (pred == y)
    acc = np.sum(same) / x.shape[0]
    return acc

if __name__ == '__main__':
    main()