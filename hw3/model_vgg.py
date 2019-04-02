import torch
import torch.nn as nn
import torch.nn.functional as F

conv_config = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    'D': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    '11':[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13':[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def make_layers(config, bn= True):
    layers = []
    in_channels = 1
    for x in config:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=(2, 2), stride= (2, 2))]
        else:
            conv2d = nn.Conv2d(in_channels, x, kernel_size= 3, padding= 1)
            relu = nn.ReLU(inplace= True)
            if bn:
                layers += [conv2d, nn.BatchNorm2d(x), relu]
            else:
                layers += [conv2d, relu]
            in_channels = x

    return nn.Sequential(*layers)

fc_config = {
    'A': [512*3*3, 512, 64, 7],
    'B': [512*3*3, 1024, 128, 7],
    'C': [512*2*2, 512, 64, 7],
    'D': [512*5*5, 512, 64, 7]
}

def make_fc(config, drop_out= True):
    layers = []
    for i in range(len(config) - 2):
        layers += [nn.Linear(config[i], config[i + 1]), nn.ReLU(inplace= True)]
        if drop_out:
            layers += [nn.Dropout()]
    layers += [nn.Linear(config[-2], config[-1])]
    
    return nn.Sequential(*layers)

class Model_VGG(nn.Module):
    def __init__(self, conv_layers, fc_layers):
        super().__init__()
        self.feature = conv_layers
        self.classifier = fc_layers
    def forward(self, inputs):
        x = self.feature(inputs)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_vgg_model(conv_id, fc_id, bn= True, drop_out= True):
    if (conv_id not in conv_config) or (fc_id not in fc_config):
        return None
    conv_layers = make_layers(conv_config[conv_id])
    fc_layers = make_fc(fc_config[fc_id])
    vgg = Model_VGG(conv_layers, fc_layers)
    return vgg

def test():
    def parameter_number(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # model = get_vgg_model('A', 'A')
    model = make_layers(conv_config['C'])
    inputs = torch.zeros(4, 1, 44, 44)
    out = model(inputs)
    print('Output size:', out.size())
    print('Parameter number:', parameter_number(model))

if __name__ == '__main__':
    test()
