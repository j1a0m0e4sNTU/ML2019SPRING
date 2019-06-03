import sys
import torch
import torch.nn as nn

# simple baseline : about 162435 parameters
# strong baseline : about 56310  parameters

conv_config = {
    'base': [16, 64, 'D', 128, 'D', 256, 'D', 512, 'D'],
    'A': [8, 16, 'D', 16, 16, 16, 'D', 32, 32, 'D', 32, 32, 'D', 64, 64],
    'B': [16, 16, 16, 'D', 16, 16, 16, 16, 'D', 32, 32, 32, 32, 'D', 64, 64, 64, 64, 'D'],
    'C': [16, 16, 16, 'D', 16, 16, 16, 'D', 32, 32, 32, 'D', 64, 64, 64, 'D', 128, 128], 
    'D': [32, 32, 32, 'D', 32, 32, 32, 'D', 64, 64, 64, 'D', 128, 128, 128, 'D', 128, 128],
    'E': [32, 32, 32, 'D', 32, 32, 32, 'D', 32, 32, 32, 'D', 64, 64, 64, 'D', 128],
    'F': [16, 16, 'D', 32, 32, 'D', 32, 32, 'D', 64, 64, 'D'],
    'G': [16, 16, 'D', 16, 16, 'D', 32, 32, 'D', 32, 32, 'D']
}

fc_config = {
    'base': [512*2*2, 128, 7],
    'A': [64*2*2, 64, 7], 
    'B': [64*2*2, 128, 7], 
    'C': [128*2*2, 7],
    'C2': [128*2*2, 32, 7],
    'F' : [64*2*2, 7], 
    'G' : [32*2*2, 7]
}

def conv_layers(config):
    layers = []
    in_planes = 1
    for i, x in enumerate(config) :
        if x == 'D':
            layers.append(nn.MaxPool2d(kernel_size= 2, stride= 2))
        else:
            layers += [
                nn.Conv2d(in_planes, x, 3, 1, 1),
                nn.BatchNorm2d(x), 
                nn.ReLU(inplace= True)
            ]
            in_planes = x

    layers = nn.Sequential(* layers)
    return layers

def fc_layers(config):
    layers = []
    for i in range(1, len(config)):
        layers += [
            nn.Linear(config[i - 1], config[i]), 
            nn.ReLU(inplace= True),
            nn.Dropout()
        ]
    layers = layers[:-2]
    layers = nn.Sequential(* layers)
    return layers

class CNN(nn.Module):
    def __init__(self, conv_symbol, fc_symbol):
        super().__init__()
        self.conv_layers = conv_layers(conv_config[conv_symbol])
        self.fc_layers = fc_layers(fc_config[fc_symbol])
    
    def forward(self, inputs):
        out = self.conv_layers(inputs)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out

def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test():
    batch_size = 8
    images = torch.zeros(batch_size, 1, 44, 44)
    model = CNN(sys.argv[1], sys.argv[2])
    output = model(images)

    print('Model parameter number: {}'.format(parameter_number(model)))
    print('Input size:  {}'.format(images.size()))
    print('Output size: {}'.format(output.size()))

if __name__ == '__main__':
    test()