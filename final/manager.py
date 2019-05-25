import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from util import *

class Manager():
    def __init__(self, model, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.load:
            model.load_state_dict(torch.load(args.load))
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr= args.lr)
        self.metric = nn.MSELoss()
        self.epoch_num = args.epoch
        self.check_epoch = args.check
        self.save = args.save
        self.output = args.output
        self.best = {'epoch':0, 'acc':0}
        self.record_file = None
        if args.record:
            self.record_file = open(args.record, 'w')
            self.record_file.write(args.info + '\n')
            self.record_file.write(' =======================\n')

    def record_info(self, info):
        self.record_file.write(info + '\n')
        print(info)

    def train(self, train_data, valid_data):
        for epoch in range(self.epoch_num):
            self.model.train()
            train_loss = 0
            train_psnr = 0
            train_ssim = 0
            for i, data in enumerate(train_data):
                hazy, gt = data
                hazy = hazy.to(self.device)
                gt   =  gt.to(self.device)
                out = self.model(hazy)
                loss = self.metric(out, gt)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_psnr += average_PSNR(out, gt)
                train_ssim += average_SSIM(out, gt)
            
            if (epoch + 1) % self.check_epoch == 0:
                batch_num = i + 1
                train_loss, train_psnr, train_ssim = train_loss/batch_num, train_psnr/batch_num, train_ssim/batch_num
                valid_loss, valid_psnr, valid_ssim = self.validate(valid_data)
                info = 'Epoch {} | train loss: {:.5f} train PSNR: {:.5f} train SSIM: {:.5f} | valid loss: {:.5f} valid PSNR: {:.5f} valid SSIM: {:.5f}'.format(
                    epoch + 1, train_loss, train_psnr, train_ssim, valid_loss, valid_psnr, valid_ssim
                )
                self.record_info(info)
    
    def validate(self, valid_data):
        self.model.eval()
        valid_loss = 0
        valid_psnr = 0
        valid_ssim = 0
        for i, data in enumerate(valid_data):
            hazy, gt = data
            hazy = hazy.to(self.device)
            gt = gt.to(self.device)
            out = self.model(hazy)
            loss = self.metric(out, gt)
            valid_loss += loss.item()
            valid_psnr += average_PSNR(out, gt)
            valid_ssim += average_SSIM(out, gt)

        batch_num = i + 1
        valid_loss, valid_psnr, valid_ssim = valid_loss/batch_num, valid_psnr/batch_num, valid_ssim/batch_num
        return valid_loss, valid_psnr, valid_ssim

    def predict(self, test_data):
        self.model.eval()
        
