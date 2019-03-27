import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_basic(nn.Module): 
    # Training   for epoch 19=> Total acc: 0.90425 | acc per class: [(0):0.885 (1):0.901 (2):0.860 (3):0.954 (4):0.880 (5):0.923 (6):0.896 ]
    # Validation for epoch 19=> Total acc: 0.48903 | acc per class: [(0):0.426 (1):0.392 (2):0.343 (3):0.655 (4):0.351 (5):0.633 (6):0.456 ]
    def __init__(self, class_num= 7):
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

        self.fc1 = nn.Linear(128*6*6, 128)
        self.fc2 = nn.Linear(128, class_num)

    def forward(self, inputs):
        x = self.conv_block(inputs)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Model_0(nn.Module): 
    # Training   for epoch 19=> Total acc: 0.92158 | acc per class: [(0):0.928 (1):0.000 (2):0.930 (3):0.961 (4):0.953 (5):0.881 (6):0.932 ]
    # Validation for epoch 19=> Total acc: 0.45803 | acc per class: [(0):0.352 (1):0.000 (2):0.375 (3):0.646 (4):0.318 (5):0.561 (6):0.436 ]
    def __init__(self, class_num= 7):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dBlock(1, 16, 4, 2, 1), 
            Conv2dBlock(16, 64, 4, 2, 1),
            Conv2dBlock(64, 128, 4, 2, 1)
        )
        self.fc1 = nn.Linear(128*6*6, 32)
        self.fc2 = nn.Linear(32, class_num)

    def forward(self, inputs):
        x = self.block(inputs)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class Model_1(nn.Module): 
    # Training   for epoch 9=> Total acc: 0.71964 | acc per class: [(0):0.678 (1):0.072 (2):0.601 (3):0.859 (4):0.696 (5):0.773 (6):0.699 ]
    # Validation for epoch 9=> Total acc: 0.49408 | acc per class: [(0):0.322 (1):0.027 (2):0.322 (3):0.762 (4):0.483 (5):0.630 (6):0.329 ]
    def __init__(self, class_num= 7):
        super().__init__()
        self.conv_block = nn.Sequential(
            ResidualBlock(1, 16, kernel_size= 4, stride= 2, padding= 1),
            ResidualBlock(16, 64, kernel_size= 4, stride= 2, padding= 1),
        )

        self.fc1 = nn.Linear(64 * 12 * 12, 16)
        self.fc2 = nn.Linear(16, class_num)

    def forward(self, inputs):
        x = self.conv_block(inputs)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Model_2(nn.Module): 
    # Training   for epoch 19=> Total acc: 0.99617 | acc per class: [(0):0.994 (1):0.994 (2):0.993 (3):0.999 (4):0.996 (5):0.996 (6):0.997 ]
    # Validation for epoch 19=> Total acc: 0.48642 | acc per class: [(0):0.354 (1):0.324 (2):0.329 (3):0.701 (4):0.380 (5):0.621 (6):0.428 ]
    def __init__(self, class_num= 7):
        super().__init__()
        self.conv_block = nn.Sequential(
            ResidualBlock(1, 16, kernel_size= 4, stride= 2, padding= 1),
            ResidualBlock(16, 64, kernel_size= 4, stride= 2, padding= 1),
            ResidualBlock(64, 64, kernel_size= 4, stride= 2, padding= 1)
        )
        self.fc = nn.Linear(64*6*6, class_num)

    def forward(self, inputs):
        x = self.conv_block(inputs)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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
