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

class basic_extractor:
    def __init__(self):
        self.feature_num = 18 * 9 + 1

    def info(self):
        return "simply use all information without transoformation"
    
    def __call__(self, data):
        feature = np.empty(self.feature_num,)
        feature[0] = 1
        feature[1:] = data.reshape(-1,)
        return feature


if __name__ == '__main__':
    data = np.zeros((18, 9))
    extractor = basic_extractor()
    features = extractor(data)
    print(features.shape)
