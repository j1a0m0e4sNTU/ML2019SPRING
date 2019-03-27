import torch
import torch.nn as nn
import torch.nn.functional as F

config = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
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

class Model_VGG(nn.Module):
    def __init__(self, feature, class_num= 7):
        super().__init__()
        self.feature = feature
        self.classifier = nn.Sequential(
            nn.Linear(512*3*3, 512),
            nn.ReLU(inplace= True),
            nn.Dropout(),
            nn.Linear(512, 64),
            nn.ReLU(inplace= True),
            nn.Dropout(),
            nn.Linear(64, class_num)
        )
    def forward(self, inputs):
        x = self.feature(inputs)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_vgg_model(name, bn= True):
    if name not in config:
        return None
    feature = make_layers(config[name])
    vgg = Model_VGG(feature)
    return vgg

def test():
    def parameter_number(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = get_vgg_model('A')
    inputs = torch.zeros(4, 1, 48, 48)
    out = model(inputs)
    print('Output size:', out.size())
    print('Parameter number:', parameter_number(model))

if __name__ == '__main__':
    test()