import os
import argparse
from PIL import Image
import torch
import torch.nn as nn
import  torchvision.transforms as transforms
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('-e', type= float, default= 1e-8)
parser.add_argument('-m', default= 'vgg16', choices=['vgg16', 'vgg19', 'resnet50', 'resnet101', 'densenet121', 'densenet169'])
parser.add_argument('-input', default= '../../data_hw5', help= 'Input image folder')
parser.add_argument('-output', default= '../../result', help= 'Output folder')
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
        img = img_data[i][0].unsqueeze(0)
        out = model(img)
        label = out.max(1)[1].squeeze().item()
        file.write('{},{}\n'.format(i, label))

def main():
    print('- main -')
    model = get_model()
    model.eval()
    img_data = MyDataset(args.input)
    loss_fn = nn.CrossEntropyLoss()    
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    
    attack_num = 0
    for i in range(len(img_data)):
        img, label = img_data[i]
        img = img.unsqueeze(0)
        img.requires_grad = True
        label = torch.LongTensor([label])

        out = model(img)
        loss = loss_fn(out, label)
        loss.backward()
        noise = img.grad
        img_noise = img.detach() + args.e * noise

        path = os.path.join(args.output, '{:0>3d}.png'.format(i))
        img = img_data.toImage(img_noise)
        img.save(path)

        out = model(img_noise)
        new_label = out.max(1)[1].item()
        new_category = img_data.get_category_name(new_label)
        origin_label = label.item()
        origin_category = img_data.get_category_name(origin_label)
        
        print('Origin label, category: ({}, {})'.format(origin_label, origin_category))
        print('New label, category:    ({}, {})'.format(new_label, new_category))
        if new_label != origin_label:
            attack_num += 1
        print('---- {}/{} -----'.format(attack_num, i + 1))

    print('attact number: {}'.format(attack_num))

def test():
    print('- test -')
    model = get_model()
    model.eval()
    img_data = MyDataset(args.input)
    attack_num = 0
    for i in range(len(img_data)):
        img, gt = img_data[i]
        img = img.unsqueeze(0)
        out = model(img)
        label = out.max(1)[1].item()
        category = img_data.get_category_name(label)
        print('{}, label {} category {}'.format(i, label, category))
        if gt != label:
            attack_num += 1
    print('attack num: {}'.format(attack_num))
    
if __name__ == '__main__':
    main()