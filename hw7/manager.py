import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import AutoEncoder
from dataset import *

class Manager():
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.id = args.id
        model = AutoEncoder(args.E, args.D)
        if args.load:
            model.load_state_dict(torch.load(args.load))
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr= args.lr)
        self.metric = nn.MSELoss()
        self.epoch_num = args.epoch
        self.batch_size = args.bs
        self.save = args.save
        self.csv = args.csv
        self.record_file = None
        self.best = {'epoch':0, 'loss': 999}
        self.test_images = get_test_image(args.dataset)

    def record(self, info):
        self.record_file.write(info + '\n')
        print(info)

    def train(self, data_train, data_valid):
        self.record_file = open('records/' + self.id + '.txt', 'w')
        for epoch in range(self.epoch_num):
            train_loss = 0
            self.model.train()
            for i, images in enumerate(data_train):
                images = images.to(self.device)
                images_out = self.model(images)
                self.optimizer.zero_grad()
                loss = self.metric(images_out, images)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
            train_loss = train_loss / (i+1)
            valid_loss = self.get_valid_loss(data_valid)
            best_info = ""
            if (valid_loss < self.best['loss']):
                self.best['epoch'] = epoch
                self.best['loss'] = valid_loss
                torch.save(self.model.state_dict(), self.save)
                best_info = '* Best *'
            info = 'Epoch {:2d} | training loss: {:.5f} | validation loss: {:.5f} {}'.format(epoch+1, train_loss, valid_loss, best_info)
            self.record(info)
            self.plot()

    def get_valid_loss(self, data_valid):
        self.model.eval()
        valid_loss = 0
        for i, images in enumerate(data_valid):
            images = images.to(self.device)
            images_out = self.model(images)
            loss = self.metric(images_out, images)
            valid_loss += loss.item()
        return valid_loss / (i+1)

    def predict(self):
        pass

    def write_submission(self, predictions):
        file = open(self.csv, 'w')
        file.write('id,label\n')
        for i, pred in enumerate(predictions):
            file.write('{},{}\n'.format(i,pred))
        print('Finish submission: {}'.format(self.csv))

    def plot(self):
        images = self.test_images.to(self.device)
        images_out = self.model(images)
        plot_images(images_out, 'records/' + self.id + '.jpg')