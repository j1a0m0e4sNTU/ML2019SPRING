import torch
import torch.nn as nn

# simple baseline : about 162435 parameters
# strong baseline : about 56310  parameters

conv_config = {
    'base': [16, 64, 'D', 128, 'D', 256, 'D', 512, 'D'],
}

fc_config = {
    'base': [512*2*2, 128, 7],
}

def conv_layers(config):
    layers = []
    in_planes = 1
    for x in config:
        if x == 'D':
            layers.append(nn.MaxPool2d(kernel_size= 2, stride= 2))
        else:
            layers += [
                MobileConv2d(in_planes, x, 3, 1, 1), 
                nn.ReLU(inplace= True)
            ]
            in_planes = x

    layers = nn.Sequential(* layers)
    return layers

def fc_layers(config):
    layers = []
    for i in range(1, len(config)):
        print(config[i - 1], config[i])
        layers += [
            nn.Linear(config[i - 1], config[i]), 
            nn.ReLU(inplace= True),
            nn.Dropout()
        ]
    layers = layers[:-2]
    layers = nn.Sequential(* layers)
    return layers

class MobileConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size= 3, stride= 1, padding= 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size, stride, padding, groups= in_planes),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace= True),

            nn.Conv2d(in_planes, out_planes, 1, 1, 0),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, inputs):
        out = self.block(inputs)
        return out

def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MobileNet(nn.Module):
    def __init__(self, conv_symbol, fc_symbol):
        super().__init__()
        self.conv_layers = conv_layers(conv_config[conv_symbol])
        self.fc_layers = fc_layers(fc_config[fc_symbol])

    def forward(self, inputs):
        out = self.conv_layers(inputs)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out

def test():
    batch_size = 8
    images = torch.zeros(batch_size, 1, 44, 44)
    model = MobileNet('base', 'base')
    output = model(images)

    print(model)
    print('Model parameter number: {}'.format(parameter_number(model)))
    print('Input size:  {}'.format(images.size()))
    print('Output size: {}'.format(output.size()))

def test2():
    in_c, out_c = 1, 32
    kernel_size = 4
    mobile = MobileConv2d(in_c, out_c, kernel_size, 1, 0)
    conv = nn.Conv2d(in_c, out_c, kernel_size, 1, 0)
    print('Parameter num of mobile block: {}'.format(parameter_number(mobile)))
    print('Parameter num of conv: {}'.format(parameter_number(conv)))


if __name__ == '__main__':
    test()