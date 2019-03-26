import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_basic(nn.Module): # Overfitting - train acc:0.8 valid acc:0.5
    def __init__(self, class_num= 6):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(inplace= True),
            nn.Conv2d(16, 64, kernel_size= 4, stride= 2, padding= 1),
            nn.ReLU(inplace= True),
            nn.Conv2d(64, 128, kernel_size= 4, stride= 2, padding= 1),
            nn.ReLU(inplace= True),
            nn.Conv2d(128, 128, kernel_size= 4, stride= 2, padding= 1),
            nn.ReLU(inplace= True)
        )

        self.fc1 = nn.Linear(128*6*6, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, class_num)

    def forward(self, inputs):
        x = self.conv_block(inputs)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class Model_1(nn.Module):
    def __init__(self, class_num= 6):
        super().__init__()
        self.conv_block = nn.Sequential(
            ResidualBlock(1, 16, kernel_size= 4, stride= 2, padding= 1),
            ResidualBlock(16, 64, kernel_size= 4, stride= 2, padding= 1),
        )

        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, class_num)

    def forward(self, inputs):
        x = self.conv_block(inputs)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size= 3, stride =1, padding= 1):
        super().__init__()
        self.downSample = None
        if stride != 1 or in_c != out_c: 
            self.downSample = Conv2dBlock(in_c, out_c, kernel_size, stride, padding, use_relu= False)

        self.residual = nn.Sequential(
            Conv2dBlock(out_c, out_c, kernel_size= 3, stride= 1, padding= 1, use_relu= True),
            Conv2dBlock(out_c, out_c, kernel_size= 3, stride= 1, padding= 1, use_relu= False)
        )

    def forward(self, inputs):
        if self.downSample:
            inputs = self.downSample(inputs)
        res = self.residual(inputs)
        x = F.relu(inputs + res)
        return x

class Conv2dBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding, use_relu= True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding= padding),
            nn.BatchNorm2d(out_c)
        )
        self.use_relu = use_relu

    def forward(self, inputs):
        x = self.block(inputs)
        if self.use_relu:
            x = F.relu(x)
        return x


def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test():
    model = Model_1()
    imgs = torch.zeros(4, 1, 48, 48)
    out = model(imgs)
    
    print('Input size:', imgs.size())
    print('Output size:', out.size())
    print('Parameter Number:', parameter_number(model))

if __name__ == '__main__':
    test()
