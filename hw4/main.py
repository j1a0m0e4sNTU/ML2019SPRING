import numpy as np
import torch
from matplotlib import pyplot as plt
import argparse
from dataset import *
from model_vgg import *

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default= '../../data_hw3/train.csv', help= 'Path to train.csv')
parser.add_argument('-load', default= 'vgg_cc_2.pkl', help= 'Path to pre-trained model weight')
args = parser.parse_args()

def get_model():
    model = get_vgg_model('C', 'C')
    model.load_state_dict(torch.load(args.load))
    return model

def get_image_for_label(dataset, label):
    for l, image in dataset:
        if l == label:
            return image

def show_image(tensor):
    tensor = tensor.squeeze()
    array = np.array(tensor)
    plt.imshow(array, cmap= 'gray')
    plt.show()

def test():
    train_set = TrainDataset(args.dataset, mode= 'train')
    i = get_image_for_label(train_set, 3)
    show_image(i)

def main():
    pass

if __name__ == '__main__':
    print('- Main -')
    test()