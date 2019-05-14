import sys
import torch
import torch.nn as nn

encoder_config = {
    'base': [3, 'down', 32, 'down', 64, 'down', 128, 'down', 256, 'down', 512],
    'A':    [3, 'down', 32, 'down', 64, 'down', 128, 'down', 128, 'down', 128],
    'B':    [3, 'down', 32, 'down', 64, 'down', 64, 'down', 64, 'down', 64],
    'C':    [3, 'down', 32, 'down', 32, 'down', 32, 'down', 32, 'down', 32],
    'D':    [3, 'down', 512, 'down', 256, 'down', 128, 'down', 64, 'down', 32],
}

decoder_config = {
    'base':[512, 'up', 256,  'up', 128, 'up', 64, 'up', 32, 'up', 3],
    'A':   [128, 'up', 128,  'up', 128, 'up', 64, 'up', 32, 'up', 3],
    'B':   [64, 'up', 64,  'up', 64, 'up', 64, 'up', 32, 'up', 3],
    'C':   [32, 'up', 32,  'up', 32, 'up', 32, 'up', 32, 'up', 3],
    'D':   [32, 'up', 64,  'up', 128, 'up', 256, 'up', 512, 'up', 3],
}

class AutoEncoder(nn.Module):
    def __init__(self, E_symbol, D_symbol):
        super().__init__()
        self.encoder = Encoder(encoder_config[E_symbol])
        self.decoder = Decoder(decoder_config[D_symbol])

    def forward(self, inputs):
        vectors = self.encoder(inputs)
        images  = self.decoder(vectors)
        return images

    def encode(self, inputs):
        vectors = self.encoder(inputs)
        return vectors

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer_count = int((len(config) - 1)/2)
        layers = []
        for i in range(layer_count):
            in_c = config[2*i]
            conv_type = config[2*i + 1]
            out_c = config[2*i + 2]
            if conv_type == 'conv':
                layers.append(nn.Conv2d(in_c, out_c, lernel_size= 3, stride= 1, padding= 1))
            elif conv_type == 'down':
                layers.append(nn.Conv2d(in_c, out_c, kernel_size= 4, stride= 2, padding= 1))
            layers += [nn.BatchNorm2d(out_c), nn.ReLU(inplace= True)]

        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        out = self.net(inputs)
        out = out.view(out.size(0), -1)
        return out

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer_count = int((len(config) - 1)/2)
        layers = []
        for i in range(layer_count):
            in_c = config[2*i]
            conv_type = config[2*i + 1]
            out_c = config[2*i + 2]
            if conv_type == 'conv':
                layers.append(nn.Conv2d(in_c, out_c, lernel_size= 3, stride= 1, padding= 1))
            elif conv_type == 'up':
                layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size= 4, stride= 2, padding= 1))
            layers += [nn.BatchNorm2d(out_c), nn.ReLU(inplace= True)]
        
        layers[-1] = nn.Sigmoid()
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        size = inputs.size()
        inputs = inputs.view(size[0], size[1], 1, 1)
        out = self.net(inputs)
        return  out

def test():
    batch_size = 8
    imgs = torch.zeros(batch_size, 3, 32, 32)
    model = AutoEncoder(sys.argv[1], sys.argv[2])
    img_out = model(imgs)
    print(img_out.size())

def test2():
    img = torch.zeros(1, 3, 1, 1)
    deconv = nn.ConvTranspose2d(3, 32, kernel_size= 4, stride= 2, padding= 1)
    out = deconv(img)
    print(out.size())

if __name__ == '__main__':
    test()