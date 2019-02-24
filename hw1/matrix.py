import numpy as np
import argparse
from util import *
from feature_extract import *

parser = argparse.ArgumentParser()
parser.add_argument('-validation', help='Validation rate', type= float, default= 0.2)
parser.add_argument('-save', help= 'Weight name to be saved', default= None)
args = parser.parse_args()

def main():
    extractor = extractor_1()
    train_data = get_aligned_train_data()
    train_x, train_y, valid_x, valid_y = get_split_data(train_data, extractor, args.validation)

    pseudo_inverse = np.linalg.inv(train_x.T @ train_x) @ (train_x.T)
    weight = pseudo_inverse @ train_y

    train_pred = train_x @  weight   
    train_error = get_mse_error(train_pred, train_y)
    print('Traing error:', train_error)
    if args.validation > 0:
        valid_pred = valid_x @ weight
        valid_error = get_mse_error(valid_pred, valid_y)
        print('Validation error:', valid_error)
    
    if args.save:
        np.save(args.save, weight)

if __name__ == '__main__':
    main()