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
        self.best = {'epoch': 0, 'acc': 0}
        self.record_file = None
        if args.record:
            self.record_file = open(args.record, 'w')
            self.record(self.model.__str__())
            self.record('--------------------')

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
                vectors = vectors.to(self.device)
                labels = labels.to(self.device) 
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
                vectors = vectors.to(self.device)
                labels = labels.to(self.device)
                labels = labels.squeeze(1)
                out = self.model(vectors)
                valid_correct += self.get_correct_num(out, labels)
            
            train_loss = train_loss / train_size
            train_acc  = train_correct / train_size
            valid_acc  = valid_correct / valid_size
            self.record('Epoch {}| Train Loss: {:.5f} | Train Acc: {:.5f} | Valid Acc: {:.5f}'.format(epoch, train_loss, train_acc, valid_acc))
            
            if valid_acc > self.best['acc']:
                self.record('* BEST SAVED *')
                self.best['epoch'] = epoch
                self.best['acc'] = valid_acc
                if self.save:
                    torch.save(self.model.state_dict(), self.save)


    def get_correct_num(self, out, labels):
        pred = self.get_prediction(out)
        labels = labels.type(torch.uint8)
        correct = torch.sum(pred == labels).item() 
        return correct 

    def get_prediction(self, out):
        pred = (torch.sign(out) == 1)
        return pred

    def get_all_predictions(self, test_data):
        predictions = []
        for i, vectors in enumerate(test_data):
            vectors = vectors.to(self.device)
            out = self.model(vectors)
            pred_list = [i for i in self.get_prediction(out)]
            predictions += pred_list
        return predictions

    def predict(self, test_data, predict_path):
        predictions = self.get_all_predictions(test_data)
        file = open(predict_path, 'w')
        file.write('id,label\n')
        for i, pred in enumerate(predictions):
            file.write('{},{}\n'.format(i, pred))
        print('Finish prediction @ {}'.format(predict_path))
