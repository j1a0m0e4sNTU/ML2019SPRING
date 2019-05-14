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
    images = np.zeros((len(image_name), 600, 600, 3))
    for i, name in enumerate(image_name):
        images[i] = np.array(io.imread(name))
    return images

def main():
    pass

def test():
    images = get_images()
    
    
if __name__ == '__main__':
    test()