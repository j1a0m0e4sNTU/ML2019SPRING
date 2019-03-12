import argparse
import numpy as np
from util import *
from extractor import *

parser = argparse.ArgumentParser()
parser.add_argument('-mode', choices= ['logistic', 'generative'], default= 'logistic', help= 'training mode')
parser.add_argument('-x', default= 'data/X_train', help= 'Path to X_train')
parser.add_argument('-y', default= 'data/Y_train', help= 'Path to Y_train')
parser.add_argument('-epoch', type= int, default= 10000, help= 'Training epoch number')
parser.add_argument('-lr', type= float, default= 1.0, help= 'Learning rate')
parser.add_argument('-check', type= int, default= 100, help= 'epoch number to check performance')
parser.add_argument('-validate', type= bool, default= True, help= 'Validate or not')
parser.add_argument('-save', default= None, help= 'Weights name')
args = parser.parse_args()

def main():
    total_x = get_raw_data(args.x)
    total_y = get_raw_data(args.y)
    extractor = extractor_basic()
    total_x = extractor(total_x)

    if args.mode == 'logistic':
        train = train_logistic
    else:
        train = train_generative
    
    if args.validate:
        train_x, train_y, valid_x, valid_y = get_train_valid_data(total_x, total_y, 0)
    else:
        train_x, train_y, valid_x, valid_y = total_x, total_y, None, None
    
    train(train_x, train_y, valid_x, valid_y)

def train_logistic(trian_x, train_y, valid_x, valid_y):
    pass

def train_generative(train_x, train_y, valid_x, valid_y):
    pass

def compute_acc(x, y, w):
    pass

def sigmoid(x):
    pass

def test():
    pass

if __name__ == '__train__':
    test()