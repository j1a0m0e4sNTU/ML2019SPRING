import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
from matplotlib import pyplot as plt
import argparse
from lime import lime_image
from skimage.segmentation import slic
from dataset import *
from model_vgg import *
torch.manual_seed(19961004)

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default= '../../data_hw3/train.csv', help= 'Path to train.csv')
parser.add_argument('-load', default= '../../vgg_cc_2.pkl', help= 'Path to pre-trained model weight')
parser.add_argument('-step', type= int, default= 50, help= 'Step number for gradient ascent')
parser.add_argument('-lr', type= float, default= 1e-1, help= 'Learning rate for gradient ascent')
args = parser.parse_args()

def get_model():
    model = get_vgg_model('C', 'C')
    model.load_state_dict(torch.load(args.load, map_location= 'cpu'))
    return model

def show_image(tensor, cmap= 'gray'):
    array = tensor.squeeze().detach().numpy()
    plt.imshow(array, cmap= cmap)
    plt.show()

def save_image(name, tensor, cmap= 'gray'):
    array = tensor.squeeze().detach().numpy()
    plt.imsave(name, array, cmap= cmap)

class HW4():
    def __init__(self, model, dataset, args):
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.loss_func = nn.CrossEntropyLoss()
        self.step = args.step
        self.lr = args.lr

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

    def get_model_part(self, conv_id):
        conv_id = conv_id % 16
        for i, layer in enumerate(self.model.feature):
            if 'Conv' in layer.__str__():
                if conv_id == 0:
                    return self.model.feature[:i+3]
                else:
                    conv_id -= 1

    def get_most_activate(self, model, filter_id):
        image = torch.rand(1, 1, 44, 44)
        image.requires_grad = True
        optimizer = optim.Adam([image], lr= self.lr)

        for _ in range(self.step):
            optimizer.zero_grad()
            out = model(image)
            filter_out = out[0, filter_id]
            loss = -torch.mean(filter_out)
            loss.backward()
            optimizer.step()
        return image

    def plot_activate_images(self, model, name, total_num= 25, shape= (5, 5)):
        for filter_id in range(total_num):
            activate_image = self.get_most_activate(model, filter_id)
            activate_image = activate_image.squeeze().detach().numpy()
            plt.subplot(shape[0], shape[1], filter_id + 1)
            plt.imshow(activate_image, cmap= 'hot')
        
        #plt.show()
        plt.savefig(name)
        plt.close()

    def plot_filter_output(self, model, image, name, total_num= 25, shape= (5, 5)):
        image = image.unsqueeze(0)
        out = model(image)
        out = out.squeeze()
        for filter_id in range(total_num):
            filter_image = out[filter_id].squeeze().detach().numpy()
            plt.subplot(shape[0], shape[1], filter_id + 1)
            plt.imshow(filter_image, cmap= 'gray')

        #plt.show()
        plt.savefig(name)
        plt.close()

    def plot_task_2(self):
        model = self.get_model_part(0)
        image = self.get_image_for_label(3, 10)
        self.plot_activate_images(model, 'fig2_1.jpg')
        self.plot_filter_output(model, image, 'fig2_2.jpg')


    def plot_task_3(self):
        def predict_fn(image):
            image = torch.from_numpy(image[:, :, :, 0]).unsqueeze(1)
            pred = self.model(image)
            pred = pred.squeeze().detach().numpy()
            return pred
        
        def segmentation_fn(image):
            image = image.astype(np.float64)
            segments = slic(image, n_segments= 100, compactness= 10)
            return segments

        explainer = lime_image.LimeImageExplainer()

        for label in range(7):
            image = self.get_image_for_label(label, 0)
            image = image.squeeze().numpy()
            explanation = explainer.explain_instance(image, classifier_fn= predict_fn, top_labels= 7, num_features= 10000, segmentation_fn= segmentation_fn)
            image, mask = explanation.get_image_and_mask(label, positive_only= False, num_features= 3, hide_rest= False)

            plt.imshow(image)
            #plt.show()
            plt.savefig('fig3_{}.jpg'.format(label))
            plt.close()

    def plot_task_4(self):
        pass

    def test(self):
        pass
        
def test():
    model = get_model()
    train_set = TrainDataset(args.dataset, mode= 'valid')
    hw4 = HW4(model, train_set, args)
    hw4.test()

def main():
    model = get_model()
    train_set = TrainDataset(args.dataset, mode= 'valid')
    hw4 = HW4(model, train_set, args)
    hw4.plot_task_1()
    hw4.plot_task_2()
    hw4.plot_task_3()

if __name__ == '__main__':
    print('- Main -')
    test()