import numpy as np
import pandas as pd
import csv
import os

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

def get_split_data(data, validation_rate= 0.2):
    train_x, train_y, valid_x, valid_y = None, None, None, None
    
    
    return train_x, train_y, valid_x, valid_y
 
    

def test():
    get_aligned_train_data()

if __name__ == '__main__':
    test()