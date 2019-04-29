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
        self.save = args.save
        self.best = {}
        if args.record:
            self.record = open(args.record, 'w')

    def train(self, train_data, valid_data):
        train_size = len(train_data)
        valid_size = len(valid_data)

        for epoch in range(self.epoch_num):
            self.model.train()
            train_correct = 0
            train_loss    = 0
            for i in range(train_size):
                vectors, label = train_data[i]
                vectors = vectors.unsqueeze(1)
                out = self.model(vectors)
                self.optimizer.zero_grad()
                loss = self.loss_function(out, label)
                loss.backward()
                self.optimizer.step()
                
                train_correct += self.evaluate(out, label)
                train_loss += loss.item()

                if (i+1) % 50 == 0:
                    print(i + 1)
            
            self.model.eval()
            valid_correct = 0
            for i in range(valid_size):
                vectors, label = valid_data[i]
                vectors = vectors.unsqueeze(1)
                out = self.model(vectors)
                valid_correct += self.evaluate(out, label)
            
            train_loss = train_loss / train_size
            train_acc  = train_correct / train_size
            valid_acc  = valid_correct / valid_size
            print('Epoch {}| Train Loss: {} | Train Acc: {} | Valid Acc: {}'.format(train_loss, train_acc, valid_acc))


    def evaluate(self, out, label):
        pred = (torch.sign(out) == 1)
        label = label.type(torch.uint8)
        correct = (pred == label) 
        return correct # 1 for correct 0 for wrong


    def predict(self, test_data, predict_path):
        pass