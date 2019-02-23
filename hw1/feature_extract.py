import numpy as np

class basic_extractor:
    def __init__(self):
        self.feature_num = 18 * 9  

    def info(self):
        return "simply use all information without transoformation"
    
    def __call__(self, data):
        f = data.reshape(-1,)
        return f


if __name__ == '__main__':
    data = np.zeros((18, 9))
    extractor = basic_extractor()
    features = extractor(data)
    print(features.shape)
