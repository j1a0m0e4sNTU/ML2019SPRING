import numpy as np
import pandas as pd
import csv
import os
from feature_extract import *

## Extract features:
def get_aligned_train_data(): 
    csv = pd.read_csv('data/train.csv', encoding='big5') # 4320 * 27
    # 18 features * 24 hr * 240 days
    data_raw = np.empty((24, 4320))
    data = np.empty((18, 24 * 240), dtype= np.float)
    
    for i in range(24):
        column = np.array(csv[str(i)])
        column[column == 'NR'] = 0
        data_raw[i] = column
    
    for i in range(240):
        data_one_day = data_raw[:, 18*i: 18*(i+1)] # (24, 18)
        data[:, 24*i : 24*(i+1)] = data_one_day.T
    
    return data #(18, 24*240)

def get_split_data(data, feature_extractor, validation_rate= 0.2):
    # input: aligned data (18, 24*240)
    # feature_extractor: used to exctract or transform features from data block with size(18, 9)
    # validation rate determines how many data are reserved for validation
    data_num = 24*240 - 9
    data_x = np.empty((data_num, feature_extractor.feature_num))
    data_y = np.empty((data_num, ))

    for i in range(data_num):
        data_block = data[:, i : i+10]
        data_x[i] = feature_extractor(data_block[:,:9])
        data_y[i] = data_block[9, 9]

    valid_num = int(data_num * validation_rate)
    train_x = data_x[valid_num:,]
    train_y = data_y[valid_num:,]
    valid_x = data_x[:valid_num,]
    valid_y = data_y[:valid_num,]

    return train_x, train_y, valid_x, valid_y

def get_test_data():
    csv = pd.read_csv('data/test.csv') # 4320 * 11
    data = np.empty((240, 18, 9))
    data_raw = np.empty((4320, 9))

    keys = ['21', '21.1', '20', '20.1', '19', '19.1', '19.2', '18', '17'] # should be hard-coded
    data_raw[0] = keys

    for i in range(len(keys)):
        column = csv[keys[i]]
        column[column == 'NR'] = 0
        data_raw[1:, i] = column

    for i in range(240):
        data_block = data_raw[18*i : 18*(i+1),:]
        data[i] = data_block

    return data #(240, 18, 9)

def get_test_feature(data, feature_extractor):
    feature = np.empty((240, feature_extractor.feature_num))
    for i in range(240):
        data_block = data[i]
        feature[i] = feature_extractor(data_block)
    return feature

def test():
    data = get_aligned_train_data()
    extractor = basic_extractor()
    train_x, train_y, valid_x, valid_y = get_split_data(data, extractor, 0)
    print(train_x.shape)
    print(train_y.shape)
    print(valid_x.shape)
    print(valid_y.shape)
    print(valid_y)

def test2():
    data = get_test_data()
    extractor = basic_extractor()
    feature = get_test_feature(data, extractor)
    print(feature)
    print(feature.shape)

if __name__ == '__main__':
    test2()