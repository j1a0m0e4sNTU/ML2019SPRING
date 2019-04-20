import os
import argparse
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import  torchvision.transforms as transforms
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('-mode', choices=['main', 'test', 'best'], default= 'main')
parser.add_argument('-e', type= float, default= 1e-2)
parser.add_argument('-m', default= 'resnet50', choices=['vgg16', 'vgg19', 'resnet50', 'resnet101', 'densenet121', 'densenet169'])
parser.add_argument('-input', default= '../../data_hw5/images', help= 'Input image folder')
parser.add_argument('-output', default= '../../result/images', help= 'Output folder')
args = parser.parse_args()

def get_model():
    model = None
    if args.m == 'vgg16':
        model = vgg16(pretrained= True)
    elif args.m == 'vgg19':
        model = vgg19(pretrained= True)
    elif args.m == 'resnet50':
        model = resnet50(pretrained= True)
    elif args.m == 'resnet101':
        model = resnet101(pretrained= True)
    elif args.m == 'densenet121':
        model = densenet121(pretrained= True)
    else:
        model = densenet169(pretrained= True) 

    model.eval()
    return model       

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
    dataset = MyDataset(args.input)
    loss_fn = nn.CrossEntropyLoss()    
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    
    total_num = len(dataset)
    for i in range(total_num):
        img, label = dataset[i]
        img = img.unsqueeze(0)
        img.requires_grad = True

        out = model(img)
        loss = loss_fn(out, label)
        loss.backward()
        noise = torch.sign(img.grad.data)
        img_noise = img.detach() + args.e * noise

        path = os.path.join(args.output, '{:0>3d}.png'.format(i))
        img = dataset.toImage(img_noise)
        img.save(path)
        
        if (i+1) %10 == 0:
            print('finished: {}/{}'.format(i+1, total_num))

def best():
    print('- best -')
    model = get_model()
    dataset = MyDataset(args.input)
    loss_fn = nn.CrossEntropyLoss()    
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    total_num = len(dataset)
    count = torch.zeros(total_num)    
    for i in range(total_num):
        img, label = dataset[i]
        img = img.unsqueeze(0) #tensor after normalized
        image = dataset.get_img_data(i).numpy().astype(np.int) #numpy row data in (3, 224, 224)

        while True:    
            img.requires_grad = True
            out = model(img)
            pred = out.max(1)[1].item()
            if pred != label:
                break
            temp = out.clone().detach()
            temp[0, pred] = -99
            target = torch.LongTensor([temp.max(1)[1].item()])
            loss = loss_fn(out, target)
            loss.backward()
            noise = torch.sign(img.grad.data)
            noise = noise.detach().squeeze().permute(1, 2, 0).numpy().astype(np.int)
            image -= noise
            image[image > 255] = 255
            image[image < 0] = 0
            img = dataset.transform(image).unsqueeze(0)

            count[i] += 1
        
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        path = os.path.join(args.output, '{:0>3d}.png'.format(i))
        image.save(path)

        if (i+1) %10 == 0:
            print('finished: {}/{}'.format(i+1, total_num))
    
    for i, c in enumerate(count):
        print('{}: {}'.format(i, c))

def test():
    print('- test -')
    model = get_model()
    img_data = MyDataset('../../result/images')
    img_data_origin = MyDataset('../../data_hw5/images')
    category = Category()

    attack_num = 0
    L_infinity = 0
    for i in range(len(img_data)):
        img, gt = img_data[i]
        gt = gt.item()
        img_input = img.unsqueeze(0)
        out = model(img_input)
        label = out.max(1)[1].item()
        
        if gt != label:
            attack_num += 1
            print('{:0>3d}/{:0>3d} | {:0>3d}.png | original: {} | predicted: {}'.format(attack_num, i+1, i, category[gt], category[label]))

        img_origin = img_data_origin.get_img_data(i)
        img_new = img_data.get_img_data(i)
        differnce = torch.abs(img_origin - img_new)
        L_infinity += torch.max(differnce).item()

    print('---------------')
    print('Attack num : {}'.format(attack_num))
    print('Attack rate: {}'.format(attack_num / len(img_data)))
    print('L infinity : {}'.format(L_infinity / len(img_data)))

if __name__ == '__main__':
    if args.mode == 'main':
        main()
    elif args.mode == 'best':
        best()
    elif args.mode == 'test':
        test()