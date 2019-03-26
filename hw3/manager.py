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
        self.epoch_num = args.epoch
        self.batch_size = args.bs
        self.save = args.save
        self.predict = args.predict

    def train(self, train_data, valid_data):
        for epoch in range(self.epoch_num):
            for step, data in enumerate(train_data):
                label, imgs = data
                label, imgs = label.to(self.device), imgs.to(self.device)
              
                out = self.model(imgs)
                self.optimizer.zero_grad()
                print(out.size())
                print(label.size())
                loss = F.nll_loss(out, label)
                loss.backward()
                self.optimizer.step()

                if (step + 1) % 10 == 0:
                    print('Epoch {} step {} | training loss: {}'.format(epoch, step + 1, loss.item()/self.batch_size))
            
            valid_acc = self.validate(valid_data)
            print('\033[1;33m Validation Acc for epoch {}:{}\033[0;37m'.format(epoch,valid_acc))

    def validate(self, valid_data):
        correct_num = 0

        for data in valid_data:
            label, imgs = data
            label, imgs = label.to(self.device), imgs.to(self.device)
            out = self.model(imgs)
            pred = torch.max(out, 1)[1]
            same = (pred == label)
            correct_num += torch.sum(same).item()

        acc = correct_num / len(valid_data)
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
            

