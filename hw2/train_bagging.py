import argparse
import os
import numpy as np
import numpy.random as random
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('-mode', default= 'train', choices= ['train', 'predict'])
parser.add_argument('-csv', default= '../data/hw2/train.csv', help= 'path to train.csv')
parser.add_argument('-x', default= '../data/hw2/X_train', help= 'Path to X_train')
parser.add_argument('-y', default= '../data/hw2/Y_train', help= 'Path to Y_train')
parser.add_argument('-n', type= int, default= 10, help= 'Number of models')
parser.add_argument('-steps', type= int, default= 500, help= 'Training step number (per model)')
parser.add_argument('-lr', type= float, default= 1e-1, help= 'Learning rate')
parser.add_argument('-regularize', type= float, default= 0, help= 'Regularization weight')
parser.add_argument('-validate', type= int, default= 1, help= 'Validate or not')
parser.add_argument('-save', default= None, help= 'Save directory')
parser.add_argument('-load', default= None, help= 'Load directory')
parser.add_argument('-sr', type= float, default= 1.0, help= 'Sample rate for each training data subset')
args = parser.parse_args()

def get_total_data():
    total_x = get_total_feature(args.csv, args.x)
    mean = np.mean(total_x, 0)
    std = np.std(total_x, 0)
    total_x = normalize_feature(total_x, mean, std)
    total_y = get_raw_data(args.y)
    total_x = add_constant_column(total_x)
    return total_x, total_y
 
class Trainer():
    def __init__(self, args):
        total_x, total_y = get_total_data()
        self.total_x = total_x
        self.total_y = total_y
        self.model_num = args.n
        self.steps = args.steps
        self.lr = args.lr
        self.regularize = args.regularize 
        self.validate = args.validate
        self.save = args.save
        if (self.save) and (not os.path.isdir(self.save)):
            os.mkdir(self.save)
        self.sample_rate = args.sr

        self.models = []

    def train(self):
        if self.validate:
            train_acc_total, valid_acc_total = 0, 0
            for fold in range(5):
                train_x, train_y, valid_x, valid_y = get_train_valid_data(self.total_x, self.total_y, fold)
                train_acc, valid_acc = self.train_on_dataset(train_x, train_y, valid_x, valid_y)
                train_acc_total += train_acc
                valid_acc_total += valid_acc
                print('Fold ', fold + 1, 'Train Acc:', train_acc, 'Validation Acc:', valid_acc, '\n')

                self.models = []

            print('---------------------------------------------')
            print('Average Train Acc:', train_acc_total/5)
            print('Average Valid Acc:', valid_acc_total/5)

        else:
            train_acc, _ =self.train_on_dataset(self.total_x, self.total_y)
            print('Training Acc:', train_acc)
    
    def train_on_dataset(self, train_x, train_y, valid_x= None, valid_y= None):
        for i in range(self.model_num):
            weight = self.train_one_model(train_x, train_y)
            pred = self.get_prediction(weight, train_x)
            acc = self.get_acc(pred, train_y)
            
            print('Modle ', i + 1, end= ' ')
            print('Training acc :', acc)
 
            self.models.append(weight)
            
        train_acc, valid_acc = None, None
        train_acc = self.aggregate_accuracy(train_x, train_y)
        if self.validate:
            valid_acc = self.aggregate_accuracy(valid_x, valid_y)
        
        return train_acc, valid_acc
            

    def train_one_model(self, train_x, train_y):
       
        data_size = train_x.shape[0]
        sample_size = int(data_size * self.sample_rate)
        sample_ids = random.choice(np.arange(data_size), size= sample_size, replace= True)
        train_x = train_x[sample_ids]
        train_y = train_y[sample_ids]

        train_x_T = train_x.T
        weight = np.zeros((train_x.shape[1]), dtype= np.float)
        grad_prev = 0

        for step in range(self.steps):
            gradient_weight = (-1) * (train_y - sigmoid(train_x @ weight))
            gradient = train_x_T @ gradient_weight + self.regularize * weight
            grad_prev += gradient ** 2
            ada = np.sqrt(grad_prev) + 1e-5
            weight -= self.lr * (gradient / ada) 

        return weight

    def get_prediction(slef, weight, x):
        prediction = sigmoid(x @ weight)
        prediction[prediction >= 0.5] = 1
        prediction[prediction <  0.5] = 0
        return prediction

    def aggregate_accuracy(self, x, y):
        model_num = len(self.models)
        data_num = x.shape[0]
        predictions = np.zeros((data_num, model_num))

        for i in range(model_num):
            weight = self.models[i]
            pred = self.get_prediction(weight, x)
            pred[pred == 0] = -1
            predictions[:, i] = pred
        
        score_aggregate = np.sum(predictions, axis= 1)
        prediction_final = np.empty((data_num))
        prediction_final[score_aggregate >= 0] = 1
        prediction_final[score_aggregate <  0] = 0
        
        acc = self.get_acc(prediction_final, y)
        return acc

    def get_acc(self, predictoin, label):
        correct = (predictoin == label)
        acc = np.sum(correct) / label.shape[0]
        return acc

def main():
    
    if args.mode == 'train':
        trainer = Trainer(args)
        trainer.train()

if __name__ == '__main__':
    main()