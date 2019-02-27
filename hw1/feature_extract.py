import numpy as np

## Data format ##
#  0 AMB_TEMP       |training err: 238.24 |validation err: 385.08
#  1 CH4            |training err: 222.93 |validation err: 367.65
#  2 CO             |training err: 203.46 |validation err: 316.41
#  3 NMHC           |training err: 205.72 |validation err: 329.99
#  4 NO             |training err: 238.79 |validation err: 360.04
#  5 NO2            |training err: 177.97 |validation err: 307.89
#  6 NOx            |training err: 186.88 |validation err: 317.21
#  7 O3             |training err: 207.59 |validation err: 339.51
#  8 PM10           |training err: 112.34 |validation err: 122.06
#  9 PM2.5          |training err:  37.19 |validation err:  43.99
# 10 RAIN_FALL      |training err: 244.10 |validation err: 385.89
# 11 RH             |training err: 228.03 |validation err: 324.65
# 12 SO2            |training err: 211.84 |validation err: 263.39
# 13 THC            |training err: 198.93 |validation err: 313.64
# 14 WD_HR          |training err: 227.56 |validation err: 378.20
# 15 WIND_DIDECT    |training err: 227.74 |validation err: 388.74
# 16 WIND_SPEED     |training err: 242.42 |validation err: 386.95
# 17 WS_HR          |training err: 245.46 |validation err: 391.53
#################
class extractor_0(): # training err: | validation err: 
    'template'
    def __init__(self):
        self.feature_num = 0 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1

        return feature

class extractor_test(): # training err: | validation err: 
    'test the relation of ith data and target'
    def __init__(self, i):
        self.feature_num = 9 + 1
        self.data_idx = i

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:] = data[self.data_idx]
        return feature

class basic_extractor: # training err: 32.2288 | validation err: 40.2216
    "simply use all information without transoformation"
    def __init__(self):
        self.feature_num = 18 * 9 + 1
    
    def __call__(self, data):
        feature = np.empty(self.feature_num,)
        feature[0] = 1
        feature[1:] = data.reshape(-1,)
        return feature

class simple_extractor(): # training err: 37.1978 | validation err: 43.9984
    'use PM2.5 data only'
    def __init__(self):
        self.feature_num = 9 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:] = data[9]
        return feature

class extractor_1(): # training err: 35.8849 | validation err: 41.6989
    'use PM10 & PM2.5'
    def __init__(self):
        self.feature_num = 18 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:10] = data[8]
        feature[10:] = data[9]
        return feature

class extractor_2(): # training err: 35.4945 | validation err: 42.3446
    'use quadratic term of PM2.5 & PM10'
    def __init__(self):
        self.feature_num = 18 * 2 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:10]  = data[8]
        feature[10:19] = data[8] ** 2
        feature[19:28] = data[9]
        feature[28:]   = data[9] ** 2
        return feature

class extractor_3(): # training err: 30.1781 | validation err: 41.0763
    'Use all data & their quadratic term'
    def __init__(self):
        self.feature_num = 18*9*2 + 1

    def __call__(self, data):
        data = data.reshape(-1)
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:163] = data
        feature[163:]  = data ** 2
        return feature

class extractor_4(): # training err: 37.0350 | validation err: 44.4270
    'use quadratic term of PM2.5'
    def __init__(self):
        self.feature_num = 9*2 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:10] = data[9]
        feature[10:]  = data[9] ** 2
        return feature

class extractor_nonlinear_1(): # training err: 31.5337 | validation err: 39.9919
    'Use all data and quadratic term of last day'
    def __init__(self):
        self.feature_num = 18*10 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:163] = data.reshape(-1)
        feature[163:]  = data[:,-1] ** 2 
        return feature

class extractor_nonlinear_2(): # training err: 31.2301 | validation err: 39.9164
    'Use all data and quadratic term of last 2 day'
    def __init__(self):
        self.feature_num = 18*11 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:163] = data.reshape(-1)
        feature[163:181]  = data[:,-1] ** 2
        feature[181:] = data[:,-2] ** 2 
        return feature

class extractor_nonlinear_3(): # training err: 31.0540 | validation err: 40.3549
    'Use all data and quadratic term of last 3 day'
    def __init__(self):
        self.feature_num = 18*12 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:163] = data.reshape(-1)
        feature[163:181]  = data[:,-1] ** 2
        feature[181:199] = data[:,-2] ** 2
        feature[199:] = data[:,-3] ** 2
        return feature

class extractor_nonlinear_4(): # training err: 30.8839 | validation err: 40.3470
    'Use all data and quadratic term of last 2 day and their product'
    def __init__(self):
        self.feature_num = 18*12 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:163] = data.reshape(-1)
        feature[163:181]  = data[:,-1] ** 2
        feature[181:199] = data[:,-2] ** 2
        feature[199:] = data[:,-1] * data[:,-2] 
        return feature

class extractor_nonlinear_5(): # training err: 30.7675 | validation err: 42.5676
    'Use all data and quadratic term of last 2 day and cubic term for the last day'
    def __init__(self):
        self.feature_num = 18*12 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:163] = data.reshape(-1)
        feature[163:181]  = data[:,-1] ** 2
        feature[181:199] = data[:,-2] ** 2
        feature[199:] = data[:,-1] ** 3
        return feature

class extractor_start_day(): 
    'use all data except for the first i day'
    def __init__(self, i):
        self.feature_num = 18*(9-i) + 1
        self.start_day = i
    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:] = data[:,self.start_day:].reshape(-1)
        return feature
    # i == 1, training err: 32.4740 | validation err: 39.9023
    # i == 2, training err: 32.6196 | validation err: 39.8294
    # i == 3, training err: 33.6547 | validation err: 40.1482 
    # i == 5, training err: 34.2184 | validation err: 40.0459

class extractor_use_feature(): 
    'Use the feature specified only'
    def __init__(self, feature_list):
        self.feature_list = feature_list
        self.feature_num = len(feature_list) * 9 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:] = data[self.feature_list,:].reshape(-1)
        return feature
    # [] 
    # | training err:  | validation err: 
    # [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] 
    # | training err: 32.3836 | validation err: 39.9097
    # [2,3,4,5,6,7,8,9,10,11,12,13,14,15] 
    # | training err: 32.5128 | validation err: 39.8197
    # [2,3,4,5,6,7,8,9,10,11,12,13] 
    # | training err: 32.6567 | validation err: 39.4657
    # [2,3,5,6,7,8,9,11,12,13] 
    # | training err: 32.7821 | validation err: 39.4311
    # [2,5,6,7,8,9,11,12,13] 
    # | training err: 32.8692 | validation err: 39.4637
    # [2,3,6,7,8,9,11,12,13] 
    # | training err: 33.0422 | validation err: 39.0832
    # [5,8,9,12] 
    # | training err: 34.6737 | validation err: 40.0748 

class extractor_final(): # training err: 32.5636 | validation err: 38.8559
    'Use all tricks'
    def __init__(self):
        self.start_day = 1
        self.feature_ids = [2,3,6,7,8,9,11,12,13]
        self.nonlinear_term = 2
        self.feature_num = (9 - self.start_day + self.nonlinear_term) * len(self.feature_ids) + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        data_selected = data[self.feature_ids, self.start_day:]
        feature[0] = 1
        linear_len = (9 - self.start_day) * len(self.feature_ids)
        feature[1 : 1+linear_len] = data_selected.reshape(-1)
        feature[1+linear_len: 1+linear_len+len(self.feature_ids)] =  data_selected[:, -1] **2
        feature[1+linear_len+len(self.feature_ids): ] = data_selected[:, -2] ** 2
        return feature

if __name__ == '__main__':
    data = np.zeros((18, 9))
    extractor = extractor_final()
    features = extractor(data)
    print(features.shape)
