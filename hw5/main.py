import os
import argparse
import torch
import  torchvision.transforms as transforms
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('-e', type= float, default= 0.01)
parser.add_argument('-m', default= 'vgg16', choices=['vgg16', 'vgg19', 'resnet50', 'resnet101', 'densenet121', 'densenet169'])
parser.add_argument('-input', default= '../../data_hw5', help= 'Input image folder')
parser.add_argument('-output', default= 'result', help= 'Output folder')
args = parser.parse_args()

def get_model():
    if args.m == 'vgg16':
        return vgg16(pretrained= True)
    elif args.m == 'vgg19':
        return vgg19(pretrained= True)
    elif args.m == 'resnet50':
        return resnet50(pretrained= True)
    elif args.m == 'resnet101':
        return resnet101(pretrained= True)
    elif args.m == 'densenet121':
        return densenet121(pretrained= True)
    else:
        return densenet169(pretrained= True)        

def generate_label():
    model = get_model()
    img_data = MyDataset(args.input)
    file = open('label.csv', 'w')
    file.write('id,label\n')
    
    for i in range(len(img_data)):
        img = img_data[i].unsqueeze(0)
        out = model(img)
        label = out.max(1)[1].squeeze().item()
        file.write('{},{}\n'.format(i, label))

def main():
    print('- main -')
    model = get_model()
    img_data = MyDataset(args.input)
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

def test():
    print('- test -')

if __name__ == '__main__':
    test()