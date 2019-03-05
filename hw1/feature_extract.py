import numpy as np

## Data format ## err : with/without
#  0 AMB_TEMP       |training err: 0.8418/0.1114 |validation err: 1.3959/0.1436
#  1 CH4            |training err: 0.7947/0.1113 |validation err: 1.3326/0.1436
#  2 CO             |training err: 0.7222/0.1116 |validation err: 1.1533/0.1436
#  3 NMHC           |training err: 0.7313/0.1113 |validation err: 1.2034/0.1439
#  4 NO             |training err: 0.8454/0.1112 |validation err: 1.3096/0.1435
#  5 NO2            |training err: 0.6313/0.1112 |validation err: 1.1172/0.1440
#  6 NOx            |training err: 0.6621/0.1112 |validation err: 1.1510/0.1438
#  7 O3             |training err: 0.7324/0.1132 |validation err: 1.2405/0.1463
#  8 PM10           |training err: 0.3995/0.1126 |validation err: 0.4373/0.1478
#  9 PM2.5          |training err: 0.1294/0.3221 |validation err: 0.1579/0.4391
# 10 RAIN_FALL      |training err: 0.8671/0.1113 |validation err: 1.4053/0.1441
# 11 RH             |training err: 0.8036/0.1114 |validation err: 1.1811/0.1448
# 12 SO2            |training err: 0.7475/0.1117 |validation err: 0.9616/0.1433
# 13 THC            |training err: 0.7097/0.1112 |validation err: 1.1387/0.1437
# 14 WD_HR          |training err: 0.8017/0.1114 |validation err: 1.3818/0.1432
# 15 WIND_DIDECT    |training err: 0.8022/0.1112 |validation err: 1.4220/0.1437
# 16 WIND_SPEED     |training err: 0.8609/0.1113 |validation err: 1.4099/0.1433
# 17 WS_HR          |training err: 0.8718/0.1112 |validation err: 1.4249/0.1437
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

class extractor_except(): # training err: | validation err: 
    'template'
    def __init__(self, i):
        self.feature_num = 17 * 9 + 1
        self.id = i
    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        count = 0
        for i in range(18):
            if i == self.id:
                continue
            feature[1 + count * 9 : 1 + (count+1) * 9] = data[i]
            count += 1
        return feature

class basic_extractor: # training err: 0.1110 | validation err: 0.1439
    "simply use all information without transoformation"
    def __init__(self):
        self.feature_num = 18 * 9 + 1
    
    def __call__(self, data):
        feature = np.empty(self.feature_num,)
        feature[0] = 1
        feature[1:] = data.reshape(-1,)
        return feature

class simple_extractor(): # training err: 0.1294 | validation err: 0.1579
    'use PM2.5 data only'
    def __init__(self):
        self.feature_num = 9 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:] = data[9]
        return feature

class extractor_1(): # training err: 0.1246 | validation err: 0.1494
    'use PM10 & PM2.5'
    def __init__(self):
        self.feature_num = 18 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:10] = data[8]
        feature[10:] = data[9]
        return feature

class extractor_2(): # training err: 0.1231 | validation err: 0.1525
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

class extractor_3(): # training err: 0.1033 | validation err: 0.1490
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

class extractor_4(): # training err: 0.1287 | validation err: 0.1603
    'use quadratic term of PM2.5'
    def __init__(self):
        self.feature_num = 9*2 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:10] = data[9]
        feature[10:]  = data[9] ** 2
        return feature

class extractor_nonlinear_1(): # training err: 0.1084 | validation err: 0.1441
    'Use all data and quadratic term of last day'
    def __init__(self):
        self.feature_num = 18*10 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:163] = data.reshape(-1)
        feature[163:]  = data[:,-1] ** 2 
        return feature

class extractor_nonlinear_2(): # training err: 0.1074 | validation err: 0.1436
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

class extractor_nonlinear_3(): # training err: 0.1066 | validation err: 0.1460
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

class extractor_nonlinear_4(): # training err: 0.1059 | validation err: 0.1463
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

class extractor_nonlinear_5(): # training err: 0.1056 | validation err: 0.1554
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
    # i == 1, training err: 0.1120 | validation err: 0.1429
    # i == 2, training err: 0.1125 | validation err: 0.1426
    # i == 3, training err: 0.1165 | validation err: 0.1433 
    # i == 4, training err: 0.1177 | validation err: 0.1432

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
    # | training err: 0.1116 | validation err: 0.1433
    # [2,3,4,5,6,7,8,9,10,11,12,13,14,15] 
    # | training err: 0.1121 | validation err: 0.1431
    # [2,3,4,5,6,7,8,9,10,11,12,13] 
    # | training err: 0.1127 | validation err: 0.1418
    # [2,3,5,6,7,8,9,10,11,12,13] 
    # | training err: 0.1128 | validation err: 0.1416
    # [2,5,6,7,8,9,10,11,12,13] 
    # | training err: 0.1132 | validation err: 0.1417
    # [2,3,6,7,8,9,11,12,13] 
    # | training err: 0.1135 | validation err: 0.1418
    # [5,8,9,12] 
    # | training err: 0.1200 | validation err: 0.1436
    # [3,5,7,8,9,10,11] 
    # | training err: 0.1150 | validation err: 0.1417
    # [2,3,5,7,8,9,10,11] 
    # | training err: 0.1144 | validation err: 0.1411
    # [2,3,5,7,8,9,10,11,12] 
    # | training err: 0.1137 | validation err: 0.1409
    # [2,3,5,7,8,9,10,11,12,13] 
    # | training err: 0.1133 | validation err: 0.1405


class extractor_final(): # training err:  | validation err: 
    'Use all tricks'
    def __init__(self):
        self.start_day = 1
        self.feature_ids = [2,3,5,7,8,9,10,11,12,13]
        self.nonlinear_term = 2
        self.feature_num = (9 - self.start_day + self.nonlinear_term) * len(self.feature_ids) + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        data_selected = data[self.feature_ids, self.start_day:]
        feature[0] = 1
        f_len = len(self.feature_ids)
        linear_len = (9 - self.start_day) * f_len
        feature[1 : 1+linear_len] = data_selected.reshape(-1)
    
        feature[(-1) * f_len :] =  data_selected[:, -1] **2
        feature[(-2) * f_len : (-1) * f_len] =  data_selected[:, -2] **2
        return feature

        # use quadratic of last 1 day: 0.1388 (r:20)
        # use quadratic of last 2 day: 0.1393 (r:20)

if __name__ == '__main__':
    data = np.zeros((18, 9))
    extractor = extractor_final()
    features = extractor(data)
    print(features.shape)
