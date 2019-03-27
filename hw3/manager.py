import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Manager():
    def __init__(self, model, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.load:
            model.load_state_dict(torch.load(args.load))
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr= args.lr)
        self.metric = nn.CrossEntropyLoss()
        self.epoch_num = args.epoch
        self.batch_size = args.bs
        self.save = args.save
        self.predict = args.predict

    def train(self, train_data, valid_data):
        for epoch in range(self.epoch_num):
            train_acc = self.get_acc_dict()
            for step, data in enumerate(train_data):
                label, imgs = data
                label, imgs = label.to(self.device), imgs.to(self.device)
                out = self.model(imgs)
                self.optimizer.zero_grad()
                loss = self.metric(out, label)
                loss.backward()
                self.optimizer.step()
                self.compute_acc(out, label, train_acc)

                if (step + 1) % 50 == 0:
                    print('Epoch {} step {} | training loss: {}'.format(epoch, step + 1, loss.item()/self.batch_size))
            
            valid_acc = self.validate(valid_data)
            print('\033[1;36m Training   for epoch {}=>\033[1;33m {}\033[0;37m'.format(epoch,self.get_acc_message(train_acc)))
            print('\033[1;36m Validation for epoch {}=>\033[1;33m {}\033[0;37m'.format(epoch,self.get_acc_message(valid_acc)))

    def validate(self, valid_data):
        valid_dcit = self.get_acc_dict()
        for data in valid_data:
            label, imgs = data
            label, imgs = label.to(self.device), imgs.to(self.device)
            out = self.model(imgs)
            self.compute_acc(out, label, valid_dcit)
        return valid_dcit

    def compute_acc(self, out, label, acc_dict):
        pred = torch.max(out, 1)[1]
        same = (pred == label)
        acc_dict['total'][0] += torch.sum(same).item()
        acc_dict['total'][1] += pred.size(0)
        for i in range(7):
            gt = (label == i)
            pr = (pred == i)
            same = ((gt + pr) == 2)
            acc_dict[i][0] += torch.sum(same).item()
            acc_dict[i][1] += torch.sum(gt).item()

    def get_acc_dict(self):
        acc_dict = {
            'total': [0, 0],
            0: [0, 0],
            1: [0, 0],
            2: [0, 0],
            3: [0, 0],
            4: [0, 0],
            5: [0, 0],
            6: [0, 0]
        }
        return acc_dict

    def get_acc_message(self, acc_dict):
        message = 'Total acc: {:.5f}'.format(acc_dict['total'][0] / acc_dict['total'][1])
        message += ' | acc per class: ['
        for i in range(7):
            message += '({}):{:.3f} '.format(i, acc_dict[i][0]/acc_dict[i][1])
        message += ']'
        return message

    def predict(self, test_data):
        file = open(self.predict)
        file.write('id, label\n')

        for i, feature in enumerate(test_data):
            feature = feature.to(self.device)
            out = self.model(feature)
            pred = torch.max(out, 1)[1].item()
            file.write(i, ',', pred)

        print('Finish prediction at', self.predict) 
            

