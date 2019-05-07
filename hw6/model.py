import sys
import torch
import torch.nn as nn

configs = {
    'simple': {'LSTM': {'input': 300, 'hidden': 512, 'layers': 1},'FC':[512, 256, 32, 1], 'bid':False},
    'A': {'LSTM': {'input': 300, 'hidden': 1024, 'layers': 1},'FC':[1024, 256, 32, 1], 'bid': False},
    'B': {'LSTM': {'input': 300, 'hidden': 512, 'layers': 1},'FC':[1024, 256, 32, 1], 'bid': True}, #bidirection
    'C': {'LSTM': {'input': 300, 'hidden': 512, 'layers': 2},'FC':[512, 256, 32, 1], 'bid':False},
}

class RNN(nn.Module):
    def __init__(self, config, batch_size):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bidirection = config['bid']
        self.lstm = nn.LSTM(
            input_size = config['LSTM']['input'],
            hidden_size = config['LSTM']['hidden'],
            num_layers = config['LSTM']['layers'],
            batch_first= True,
            bidirectional = bidirection
        )

        hid_layer = config['LSTM']['layers'] * 2 if bidirection else config['LSTM']['layers']
        self.state_shape = (hid_layer, batch_size, config['LSTM']['hidden'])

        fc_layers = []
        for i in range(len(config['FC']) - 1):
            fc_layers.append(nn.Linear(config['FC'][i], config['FC'][i + 1]))
            fc_layers.append(nn.ReLU(inplace= True))
        fc_layers.pop()
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, inputs):
        # inputs size: (batch_size, sequence_len, feature_number)
        # output size of LSTM: (batch_size, sequence_len,, hidden_size)
        states = (torch.zeros(self.state_shape).to(self.device), torch.zeros(self.state_shape).to(self.device))
        r_out, new_states = self.lstm(inputs, states)
        last_out = r_out[:, -1, :].squeeze(1)
        out = self.classifier(last_out)
        return out.view(-1)

def get_rnn_model(name, batch_size):
    model = RNN(configs[name], batch_size)
    return model

def test_lstm():
    rnn = nn.LSTM(10, 20, 1)
    inputs = torch.zeros(30, 3, 10)
    h0 = torch.zeros(1, 3, 20)
    c0 = torch.zeros(1, 3, 20)
    output, (hn, cn) = rnn(inputs, (h0, c0))

    print(output.size())
    print(hn.size())
    print(cn.size())

def test():
    print('- test -')
    batch_size = 8
    rnn = get_rnn_model(sys.argv[1], batch_size)
    inputs = torch.zeros(batch_size, 30, 300)
    out = rnn(inputs)
    print(out.size())
    
if __name__ == '__main__':
    test()
