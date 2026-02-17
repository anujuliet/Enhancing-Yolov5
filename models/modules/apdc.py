# models/modules/apdc.py
import torch
import torch.nn as nn

class APDC(nn.Module):
    """
    Adaptive Pyramid Dilated Convolution (APDC)
    Used as a standalone mid-level context module
    """
    def __init__(self, channels):
        super().__init__()

        self.d1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1, bias=False)
        self.d2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False)
        self.d3 = nn.Conv2d(channels, channels, 3, padding=3, dilation=3, bias=False)

        self.fuse = nn.Conv2d(channels * 3, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)

        out = torch.cat([x1, x2, x3], dim=1)
        out = self.fuse(out)
        out = self.bn(out)
        return self.act(out + x)  # residual
