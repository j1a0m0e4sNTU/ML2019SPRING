import argparse
import numpy as np
from util import *
from extractor import *

parser = argparse.ArgumentParser()
parser.add_argument('-train_csv', default= 'data/train.csv', help= 'path to train.csv')
parser.add_argument('-train_f', default= 'data/X_train', help= 'path to X_train')
parser.add_argument('-test_csv', default= 'data/test.csv', help= 'path to test.csv')
parser.add_argument('-test_f', default= 'data/X_test', help= 'path to X_test')
parser.add_argument('-load', default= None, help= 'Weight.npy')
parser.add_argument('-save', default= None, help= 'prediction.csv')
parser.add_argument('-th', type= float, default= 0.5, help= 'threshold to determine 0/1')
args = parser.parse_args()

def main():
    extractor = extractor_basic()
    train_total = get_total_feature(args.train_csv, args.train_f)
    mean = np.mean(train_total, 0)
    std = np.std(train_total, 0)
    test_total = get_total_feature(args.test_csv, args.test_f)
    test_x = normalize_feature(test_total, mean, std)

    test_x = extractor(test_x)
    weight = np.load(args.load)

    prediction = sigmoid(test_x @ weight)
    prediction[prediction > args.th] = 1
    prediction[prediction <=args.th] = 0
    num = test_x.shape[0]

    file = open(args.save, 'w')
    file.write('id,label\n')
    for i in range(num):
        line = str(i + 1) + ',' + str(int(prediction[i])) + '\n'
        file.write(line)
    
    print('Prediction complete !')


if __name__ == '__main__':
    main()