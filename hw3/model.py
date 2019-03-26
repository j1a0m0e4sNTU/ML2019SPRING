import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_basic(nn.Module):
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

    def forward(self, imgs):
        x = self.conv_block(imgs)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test():
    model = Model_basic()
    imgs = torch.zeros(4, 1, 48, 48)
    out = model(imgs)
    
    print('Input size:', imgs.size())
    print('Output size:', out.size())
    print('Parameter Number:', parameter_number(model))

if __name__ == '__main__':
    test()