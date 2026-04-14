# models/modules/pea.py
import torch.nn as nn


class PEA(nn.Module):
    """Position and Edge Attention (PEA) Lightweight spatial attention.
    """

    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.dwconv(x))
        return x * attn
