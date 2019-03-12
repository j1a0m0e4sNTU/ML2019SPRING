import numpy as np

class extractor_basic():
    def __init__(self):
        self.dim = 1 + 106
    
    def __call__(self, inputs):
        num, _ = inputs.shape
        feature = np.empty((num, self.dim), dtype= np.float)
        feature[:, 0] = 1
        feature[:, 1:] = inputs
        return feature