import torch
import torch.nn as nn

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

def test():
    batch_size = 8
    images = torch.zeros(batch_size, 3, 48, 48)
    model = MobileConv2d(3, 16, 4, 2, 1)
    output = model(images)

    print(model)
    print('Model parameter number: {}'.format(parameter_number(model)))
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