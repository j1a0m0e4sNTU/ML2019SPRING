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

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        planes = [16, 32, 64, 128, 256, 512]
        self.conv1 = ConvBlock(3, planes[0], 3, 1)
        self.conv2 = ConvBlock(planes[0], planes[1], 4, 2)
        self.conv3 = ConvBlock(planes[1], planes[2], 4, 2)
        self.conv4 = ConvBlock(planes[2], planes[3], 4, 2)
        self.conv5 = ConvBlock(planes[3], planes[4], 4, 2)
        self.conv6 = ConvBlock(planes[4], planes[5], 4, 2)
        self.deconv1 = DeconvBlock(planes[5], planes[4], 4, 2)
        self.deconv2 = DeconvBlock(planes[4], planes[3], 4, 2)
        self.deconv3 = DeconvBlock(planes[3], planes[2], 4, 2)
        self.deconv4 = DeconvBlock(planes[2], planes[1], 4, 2)
        self.deconv5 = DeconvBlock(planes[1], planes[0], 4, 2)
        self.deconv6 = DeconvBlock(planes[0], 3, 3, 1, relu= False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        down1 = self.conv1(inputs)
        down2 = self.conv2(down1)
        down3 = self.conv3(down2)
        down4 = self.conv4(down3)
        down5 = self.conv5(down4)
        down6 = self.conv6(down5)
        up1 = self.deconv1(down6)
        up2 = self.deconv2(down5 + up1)
        up3 = self.deconv3(down4 + up2)
        up4 = self.deconv4(down3 + up3)
        up5 = self.deconv5(down2 + up4)
        up6 = self.deconv6(down1 + up5)
        out = self.sigmoid(up6)
        return out

def test():
    inputs = torch.zeros(8, 3, 512, 512)
    model = Unet()
    out = model(inputs)
    print('Input size:  {}'.format(inputs.size()))
    print('Output size: {}'.format(out.size()))

if __name__ == '__main__':
    test()