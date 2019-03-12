import argparse
import numpy as np
from util import *
from extractor import *

parser = argparse.ArgumentParser()
parser.add_argument('-source', default= 'data/X_test', help= 'test_x file')
parser.add_argument('-load', default= None, help= 'Weight.npy')
parser.add_argument('-save', default= None, help= 'prediction.csv')
args = parser.parse_args()

def main():
    extractor = extractor_basic()
    test_x = get_raw_data(args.source)
    test_x = extractor(test_x)
    weight = np.load(args.load)

    prediction = test_x @ weight
    prediction[prediction > 0.5] = 1
    prediction[prediction <= 0.5] = 0
    num = test_x.shape[0]

    file = open(args.save, 'w')
    file.write('id,label\n')
    for i in range(num):
        line = str(i + 1) + ',' + str(int(prediction[i])) + '\n'
        file.write(line)
    
    print('Prediction complete !')


if __name__ == '__main__':
    main()