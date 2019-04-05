import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import argparse
from dataset import *
from model_vgg import *

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default= '../../data_hw3/train.csv', help= 'Path to train.csv')
parser.add_argument('-load', default= '../../vgg_cc_2.pkl', help= 'Path to pre-trained model weight')
args = parser.parse_args()

def get_model():
    model = get_vgg_model('C', 'C')
    model.load_state_dict(torch.load(args.load, map_location= 'cpu'))
    return model

def show_image(tensor, cmap= 'gray'):
    array = tensor.squeeze().numpy()
    plt.imshow(array, cmap= cmap)
    plt.show()

def save_image(name, tensor, cmap= 'gray'):
    array = tensor.squeeze().numpy()
    plt.imsave(name, array, cmap= cmap)

class HW4():
    def __init__(self, model, dataset):
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.loss_func = nn.CrossEntropyLoss()

    def get_image_for_label(self, label, count= 0):
        c = 0
        for l, image in self.dataset:
            if l == label:
                if c == count:
                    image = image[:, 2:46, 2:46]
                    return image
                else:
                    c += 1

    def get_salience_map(self, label, image):
        label = torch.LongTensor([label])
        image = image.unsqueeze(0)
        image.requires_grad = True
        out = self.model(image)
        loss = self.loss_func(out, label)
        loss.backward()
        salience_map = image.grad.abs()
        return salience_map

    def plot_task_1(self):
        for label in range(7):
            image = self.get_image_for_label(label, 15)
            #save_image('{}.jpg'.format(label), image)
            salience_map = self.get_salience_map(label, image)
            save_image('fig1_{}.jpg'.format(label), salience_map, cmap= 'hot')

    def test(self):
        label = 3
        image = self.get_image_for_label(label, 15)
        show_image(image)
        salience_map = self.get_salience_map(label, image)
        save_image('test.png',salience_map, 'hot')



def test():
    model = get_model()
    train_set = TrainDataset(args.dataset, mode= 'valid')
    hw4 = HW4(model, train_set)
    hw4.test()

def main():
    model = get_model()
    train_set = TrainDataset(args.dataset, mode= 'valid')
    hw4 = HW4(model, train_set)
    hw4.plot_task_1()

if __name__ == '__main__':
    print('- Main -')
    main()