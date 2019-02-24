import numpy as np
import argparse
from util import *
from feature_extract import *

parser = argparse.ArgumentParser()
parser.add_argument('-load', help= 'Weight to be loaded', default= None)
parser.add_argument('-submit', help= 'Submission file name', default= 'submission.csv')
args = parser.parse_args()

def predict():
    test_data = get_test_data()
    extractor = basic_extractor()
    test_feature = get_test_feature(test_data, extractor)
    weight = np.load(args.load)
    test_ans = test_feature @ weight
    
    file = open(args.submit, 'w')
    file.write('id,value\n')
    for i in range(240):
        file.write('id_'+str(i)+','+str(test_ans[i])+'\n')
    
    print('Predictoin compelte !')

if __name__ == '__main__':
    predict()