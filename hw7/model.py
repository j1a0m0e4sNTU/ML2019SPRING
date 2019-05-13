import torch
import torch.nn as nn

encoder_config = {
    'base':[3, 'down', 32, 'down', 64, 'down', 128, 'down', 256, 'down', 512]
}

decoder_config = {
    'base':[512, 'up', 256,  'up', 128, 'up', 64, 'up', 32, 'up', 3]
}

def get_encoder_decoder(symbol_E, symbol_D):
    encoder = Encoder(encoder_config[symbol_E])
    decoder = Decoder(decoder_config[symbol_D])
    return encoder, decoder

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
    encoder, decoder = get_encoder_decoder('base', 'base')
    vector = encoder(imgs)
    img_out = decoder(vector)
    print(decoder)
    print(img_out)
    print(img_out.size())

def test2():
    img = torch.zeros(1, 3, 1, 1)
    deconv = nn.ConvTranspose2d(3, 32, kernel_size= 4, stride= 2, padding= 1)
    out = deconv(img)
    print(out.size())

if __name__ == '__main__':
    test()