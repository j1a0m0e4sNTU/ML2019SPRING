import numpy as np

## Data format ##
#  0 AMB_TEMP
#  1 CH4
#  2 CO
#  3 NMHC
#  4 NO
#  5 NO2
#  6 NOx
#  7 O3
#  8 PM10
#  9 PM2.5
# 10 RAIN_FALL
# 11 RH
# 12 SO2
# 13 THC
# 14 WD_HR
# 15 WIND_DIDECT
# 16 WIND_SPEED
# 17 WS_HR
#################
class extractor_0():
    'template'
    def __init__(self):
        self.feature_num = 0 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1

        return feature

class basic_extractor:
    "simply use all information without transoformation"
    def __init__(self):
        self.feature_num = 18 * 9 + 1
    
    def __call__(self, data):
        feature = np.empty(self.feature_num,)
        feature[0] = 1
        feature[1:] = data.reshape(-1,)
        return feature

class simple_extractor():
    'use PM2.5 data only'
    def __init__(self):
        self.feature_num = 9 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:] = data[9]
        return feature

class extractor_1():
    'use PM10 & PM2.5'
    def __init__(self):
        self.feature_num = 18 + 1

    def __call__(self, data):
        feature = np.empty(self.feature_num)
        feature[0] = 1
        feature[1:10] = data[8]
        feature[10:] = data[9]
        return feature

if __name__ == '__main__':
    data = np.zeros((18, 9))
    extractor = extractor_1()
    features = extractor(data)
    print(features.shape)
