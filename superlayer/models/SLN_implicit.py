import sys
import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .unet_parts import simple_block


class SLNet(nn.Module):
    
    def __init__(self, input_ch, out_ch, use_bn, superblock_size, depth):
        super(SLNet, self).__init__()
        half_size = int(superblock_size/2)
        
        self.depth = depth
        self.n_classes = out_ch
        self.down = torch.nn.MaxPool2d(2,2)

        self.block0 = simple_block(input_ch , half_size, use_bn)   
        self.down_block = simple_block(half_size, half_size, use_bn)
        self.up_block = simple_block(superblock_size, half_size, use_bn)

        self.out_conv = nn.Conv2d(half_size, out_ch, kernel_size=3, padding=1)
        self.sm = nn.Softmax(dim=1)

        
    def forward(self, x_in):

        #Model
        enc_seq = [self.block0(x_in)]
        
        for i in range(self.depth-1):
            x = self.down_block(self.down(enc_seq[-1]))
            enc_seq.append(x)
        
        x = self.down_block(self.down(enc_seq[-1]))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        for i in range(self.depth):
            x = torch.cat([x, enc_seq[-(i+1)]], 1)
            x = self.up_block(x)
            if i < (self.depth - 1):
                x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        out = self.out_conv(x)
        out = self.sm(out)
        
        return out