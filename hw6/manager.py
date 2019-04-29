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
        self.loss_function = nn.BCEWithLogitsLoss()
        self.epoch_num = args.epoch
        self.batch_size = args.batch_size
        self.save = args.save
        self.best = {}
        self.record_file = None
        if args.record:
            self.record_file = open(args.record, 'w')
            self.record_file.write(self.model.__str__())
            self.record_file.write('--------------------')

    def record(self, info):
        print(info)
        if self.record_file:
            self.record_file.write(info + '\n')
        
    def train(self, train_data, valid_data):
        train_size = len(train_data) * self.batch_size
        valid_size = len(valid_data) * self.batch_size

        for epoch in range(self.epoch_num):
            self.model.train()
            train_correct = 0
            train_loss    = 0
            for i, (vectors, labels) in enumerate(train_data):
                
                labels = labels.squeeze(1)
                out = self.model(vectors)
                self.optimizer.zero_grad()
                loss = self.loss_function(out, labels)
                loss.backward()
                self.optimizer.step()
                
                train_correct += self.get_correct_num(out, labels)
                train_loss += loss.item()
            
            self.model.eval()
            valid_correct = 0
            for i, (vectors, labels) in enumerate(valid_data):
                labels = labels.squeeze(1)
                out = self.model(vectors)
                valid_correct += self.get_correct_num(out, labels)
            
            train_loss = train_loss / train_size
            train_acc  = train_correct / train_size
            valid_acc  = valid_correct / valid_size
            self.record('Epoch {}| Train Loss: {} | Train Acc: {} | Valid Acc: {}'.format(train_loss, train_acc, valid_acc))
            if self.save:
                torch.save(self.model.state_dict(), self.save)


    def get_correct_num(self, out, labels):
        pred = (torch.sign(out) == 1)
        labels = labels.type(torch.uint8)
        correct = torch.sum(pred == labels).item() 
        return correct 


    def predict(self, test_data, predict_path):
        pass