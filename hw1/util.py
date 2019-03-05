import numpy as np
import pandas as pd
import csv
import os
from feature_extract import *

## Extract features:
def get_aligned_train_data(): 
    csv = pd.read_csv('data/train.csv', encoding='big5') # 4320 * 27
    # (18 features * 20 days * 12 months) * 24 hr
    
    data_csv = np.empty((18 * 20 * 12, 24))
    for i in range(24):
        column = csv[str(i)]
        column[column == 'NR'] = 0
        data_csv[:, i] = column
    
    data = np.empty((18, 24 * 20 * 12), dtype = np.float)
    for i in range(20 * 12):
        data[:, i * 24: (i + 1) * 24] = data_csv[i * 18: (i + 1) * 18, :]
    
    return data #(18, 24 * 20 * 12)

def normalize(data, mean, std):
    # data (18, x) mean (18,) std (18, )
    mean, std = mean.reshape(-1, 1), std.reshape(-1, 1)
    data_normal = (data - mean) / std
    return data_normal

def get_split_data(data, feature_extractor, validation_rate= 0.2):
    # input: normalized data (18, 24 * 20 * 12)
    # feature_extractor: used to exctract or transform features from data block with size(18, 9)
    # validation rate determines how many data are reserved for validation
    num_per_month = 24 * 20 - 9
    data_x = np.empty((num_per_month * 12, feature_extractor.feature_num))
    data_y = np.empty((num_per_month * 12, ))

    for month in range(12):
        month_start = 24 * 20 * month
        for i in range(num_per_month):
            data_block = data[:, month_start + i : month_start + i + 10]
            data_x[num_per_month * month + i] = feature_extractor(data_block[:, :-1])
            data_y[num_per_month * month + i] = data_block[:, -1][9]

    valid_num = int(num_per_month * 12 * validation_rate)
    train_x = data_x[valid_num:,]
    train_y = data_y[valid_num:,]
    valid_x = data_x[:valid_num,]
    valid_y = data_y[:valid_num,]

    return train_x, train_y, valid_x, valid_y

def get_predict_data(mean, std):
    csv = pd.read_csv('data/test.csv') # 4320 * 11
    data_raw = np.empty((240 * 18, 9))
    keys = ['21', '21.1', '20', '20.1', '19', '19.1', '19.2', '18', '17'] # should be hard-coded
    data_raw[0] = keys
    for i in range(len(keys)):
        column = csv[keys[i]]
        column[column == 'NR'] = 0
        data_raw[1:, i] = column
    
    data_aligned = np.empty((18, 240 * 9), dtype= np.float)
    for i in range(240):
        data_aligned[:, i * 9:(i+1) * 9] = data_raw[i * 18:(i+1) * 18, :]

    data_normal = normalize(data_aligned, mean, std)

    data = np.empty((240, 18, 9), dtype= np.float)
    for i in range(240):
        data[i] = data_normal[:, i * 9: (i+1) * 9]

    return data #(240, 18, 9)

def get_predict_feature(data, feature_extractor):
    feature = np.empty((240, feature_extractor.feature_num))
    for i in range(240):
        data_block = data[i]
        feature[i] = feature_extractor(data_block)
    return feature


def get_mse_error(x, y):
    # get mean square error
    num = x.shape[0]
    mse = np.sum(np.square(x - y))/num
    return mse

def test_train():
    data_train = get_aligned_train_data()
    mean = np.mean(data_train, 1)
    std  = np.std(data_train, 1)
    data_train = normalize(data_train, mean, std)
   
    extractor = basic_extractor()
    train_x, train_y, valid_x, valid_y = get_split_data(data_train, extractor)
    print(train_x.shape)
    print(train_y.shape)
    print(valid_x.shape)
    print(valid_y.shape)

def test_predict():
    data_train = get_aligned_train_data()
    mean = np.mean(data_train, 1)
    std  = np.std(data_train, 1)
    predict_data = get_predict_data(mean, std)
    print(predict_data.shape)

    extractor = basic_extractor()
    predict_feature = get_predict_feature(predict_data, extractor)
    print(predict_feature)

if __name__ == '__main__':
    test_train()