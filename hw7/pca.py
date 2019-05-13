import os
import argparse
import numpy as np
from skimage import io

parser = argparse.ArgumentParser()
parser.add_argument('-path', help= 'Path to data folder', default= '../../data_pca')
args = parser.parse_args()


def get_images():
    image_name = [os.path.join(args.path, name) for name in os.listdir(args.path)]
    image_name.sort()
    images = [np.array(io.imread(name)) for name in image_name]
    images = np.array(images).astype(np.float)
    return images

def main():
    pass

def test():
    pass

if __name__ == '__main__':
    test()