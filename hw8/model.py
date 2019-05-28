import torch
import torch.nn as nn



def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test():
    
    model = None

if __name__ == '__main__':
    test()