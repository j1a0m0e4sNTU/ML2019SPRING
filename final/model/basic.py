import torch
import torch.nn as nn

def ConvBlock(in_planes, out_planes, kernel_size= 3, stride= 1, relu= True):
    padding = (kernel_size - stride) // 2
    layers = [
        nn.Conv2d(in_planes, out_planes, kernel_size= kernel_size, stride= stride, padding=  padding),
        nn.BatchNorm2d(out_planes)
    ]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    block = nn.Sequential(*layers)
    return block

def DeconvBlock(in_planes, out_planes, kernel_size= 4, stride= 2, relu= True):
    padding = (kernel_size - stride) // 2
    layers = [
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size= kernel_size, stride= stride, padding= padding),
        nn.BatchNorm2d(out_planes)
    ]
    if relu:
        layers.append(nn.ReLU(inplace= True))
    block = nn.Sequential(* layers)
    return block

class Basic(nn.Module):
    def __init__(self):
        super().__init__()
        planes = [32, 64, 128, 256, 512, 1024]
        self.net = nn.Sequential(
            ConvBlock(3, planes[0], 3, 1),
            ConvBlock(planes[0], planes[1], 4, 2),
            ConvBlock(planes[1], planes[2], 4, 2),
            ConvBlock(planes[2], planes[3], 4, 2),
            ConvBlock(planes[3], planes[4], 4, 2),
            ConvBlock(planes[4], planes[5], 4, 2),
            DeconvBlock(planes[5], planes[4], 4, 2),
            DeconvBlock(planes[4], planes[3], 4, 2),
            DeconvBlock(planes[3], planes[2], 4, 2),
            DeconvBlock(planes[2], planes[1], 4, 2),
            DeconvBlock(planes[1], planes[0], 4, 2),
            DeconvBlock(planes[0], 3, 3, 1, relu= False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        out = self.net(inputs)
        return out

def test():
    inputs = torch.zeros(8, 3, 512, 512)
    model = Basic()
    out = model(inputs)
    print('Input size:  {}'.format(inputs.size()))
    print('Output size: {}'.format(out.size()))

if __name__ == '__main__':
    test()