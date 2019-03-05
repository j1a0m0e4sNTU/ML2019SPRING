import numpy as np
import argparse
from util import *
from feature_extract import *

parser = argparse.ArgumentParser()
parser.add_argument('-source', help= 'source to test.csv', default= 'data/test.csv')
parser.add_argument('-load', help= 'Weight to be loaded', default= None)
parser.add_argument('-submit', help= 'Submission file name', default= 'submission.csv')
parser.add_argument('-mode', help= 'train/matrix')
args = parser.parse_args()

def predict():
    extractor = None
    if args.mode == 'train':
        extractor = basic_extractor()
    else:
        extractor = extractor_final()
    
    data_train = get_aligned_train_data()
    mean = np.mean(data_train, 1)
    std  = np.std(data_train, 1)
    predict_data = get_predict_data(mean, std, args.source)
    predict_feature = get_predict_feature(predict_data, extractor)
    
    weight = np.load(args.load)
    predict_ans = predict_feature @ weight
    predict_ans = (predict_ans * std[9]) + mean[9]

    file = open(args.submit, 'w')
    file.write('id,value\n')
    for i in range(240):
        file.write('id_'+str(i)+','+str(predict_ans[i])+'\n')

    print('Predictoin compelte !')

if __name__ == '__main__':
    predict()