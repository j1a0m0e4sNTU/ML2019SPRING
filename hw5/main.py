import os
import argparse
import torch
import  torchvision.transforms as transforms
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('-e', type= float, default= 0.01)
parser.add_argument('-input', default= '../../data_hw5', help= 'Input image folder')
parser.add_argument('-output', default= 'result', help= 'Output folder')
args = parser.parse_args()

def main():
    print('- main -')

def test():
    print('- test -')
    model = vgg16(pretrained= True)
    img_data = MyDataset(args.input)
    img = img_data[25].unsqueeze(0)
    out = model(img)
    print(out.max(1)[1])

def generate_label():
    model = vgg16(pretrained= True)
    img_data = MyDataset(args.input)
    file = open('label.csv', 'w')
    file.write('id,label\n')
    
    for i in range(len(img_data)):
        img = img_data[i].unsqueeze(0)
        out = model(img)
        label = out.max(1)[1].squeeze().item()
        file.write('{},{}\n'.format(i, label))

if __name__ == '__main__':
    generate_label()