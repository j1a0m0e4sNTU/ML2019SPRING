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
            train_correct = 0
            train_total = 0
            for step, data in enumerate(train_data):
                label, imgs = data
                label, imgs = label.to(self.device), imgs.to(self.device)
              
                out = self.model(imgs)
                self.optimizer.zero_grad()
                loss = self.metric(out, label)
                loss.backward()
                self.optimizer.step()
                
                pred = torch.max(out, 1)[1]
                same = (pred == label)
                train_correct += torch.sum(same).item()
                train_total += pred.size(0)

                if (step + 1) % 50 == 0:
                    print('Epoch {} step {} | training loss: {}'.format(epoch, step + 1, loss.item()/self.batch_size))
            
            valid_acc = self.validate(valid_data)
            print('\033[1;33m Average Training Acc for epoch {}: {}\033[0;37m'.format(epoch,train_correct/train_total))
            print('\033[1;33m Validation Acc for epoch {}: {}\033[0;37m'.format(epoch,valid_acc))

    def validate(self, valid_data):
        correct_num = 0
        total_num = 0
        for data in valid_data:
            label, imgs = data
            label, imgs = label.to(self.device), imgs.to(self.device)
            out = self.model(imgs)
            pred = torch.max(out, 1)[1]
            same = (pred == label)
            correct_num += torch.sum(same).item()
            total_num += pred.size(0)

        acc = correct_num / total_num
        return acc

    def predict(self, test_data):
        file = open(self.predict)
        file.write('id, label\n')

        for i, feature in enumerate(test_data):
            feature = feature.to(self.device)
            out = self.model(feature)
            pred = torch.max(out, 1)[1].item()
            file.write(i, ',', pred)

        print('Finish prediction at', self.predict) 
            

