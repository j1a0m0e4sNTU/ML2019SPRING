import torch
import torch.nn as nn

configs = {
    'simple': {'LSTM': {'input': 300, 'hidden': 512, 'layers': 1},'FC':[512, 256, 32, 1]}
}

class RNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = config['LSTM']['input'],
            hidden_size = config['LSTM']['hidden'],
            num_layers = config['LSTM']['layers']
        )
        self.state_shape = (config['LSTM']['layers'], 1, config['LSTM']['hidden'])

        fc_layers = []
        for i in range(len(config['FC']) - 1):
            fc_layers.append(nn.Linear(config['FC'][i], config['FC'][i + 1]))
            fc_layers.append(nn.ReLU(inplace= True))
        fc_layers.pop()
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, inputs):
        # inputs size: (sequence_len, batch_size, feature_number)
        # output size of LSTM: (sequence_len, batch_size, hidden_size)
        states = (torch.zeros(self.state_shape), torch.zeros(self.state_shape))
        r_out, new_states = self.lstm(inputs, states)
        last_out = r_out[-1, :, :]
        out = self.classifier(last_out)
        return out.view(-1)

def get_rnn_model(name):
    model = RNN(configs[name])
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
    rnn = get_rnn_model('simple')
    inputs = torch.zeros(45, 1, 300)
    out = rnn(inputs)
    print(out.size())
    
if __name__ == '__main__':
    test()