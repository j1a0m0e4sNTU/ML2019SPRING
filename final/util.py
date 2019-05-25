import numpy as np
import torch

def PSNR(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    psnr = 10 * torch.log10(1 / mse)
    return psnr

def SSIM(img1, img2):
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = torch.sqrt(((img1 - mu1) ** 2).mean())
    sigma2 = torch.sqrt(((img2 - mu2) ** 2).mean())
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 1
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim

def test():
    img1 = torch.ones(3, 100, 100)
    img2 = torch.ones(3, 100, 100) * 0.9
    print('PSNR : {}'.format(PSNR(img1, img2)))
    print('SSIM : {}'.format(SSIM(img1, img2))) 

if __name__ == '__main__':
    test()