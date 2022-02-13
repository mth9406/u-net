from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class Conv2dBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv2dBlock, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (3,3), padding= 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, (3,3), padding= 1),
                nn.BatchNorm2d(out_channels)
        )
        self.mapping = nn.Conv2d(in_channels, out_channels, (1,1))
        
    def forward(self, x):
        out = self.block(x)
        out = out + self.mapping(x)
        out = F.relu(out)
        return out