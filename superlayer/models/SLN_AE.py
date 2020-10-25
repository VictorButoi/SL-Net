import sys
import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.distributions.normal import Normal

from .unet_parts import conv_block
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class SLN_AE(nn.Module):
    
    def __init__(self, input_ch, out_ch, use_bn, superblock_size, depth, W=None, b=None, conv_num=None, ignore_last=False):
        super(SLN_AE, self).__init__()
        dim = 2
        
        if conv_num is None:
            self.conv_repeats = [1]*(2 * depth)
        else:
            self.conv_repeats = conv_num
        
        self.depth = depth
        self.n_classes = out_ch
        self.down = torch.nn.MaxPool2d(2,2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        #Kernel size is 3
        if W is None:
            nd = Normal(0, 1e-5) 
            self.W = nn.Parameter(nd.sample((superblock_size, superblock_size,3,3)))
            self.b = nn.Parameter(torch.zeros(superblock_size))  
        else:
            self.W = W

        self.in_conv = conv_block(dim, input_ch, superblock_size, train=True)  
        
        self.super_block = conv_block(dim, train=False)
        
        self.out_conv = nn.Conv2d(superblock_size, out_ch, kernel_size=3, padding=1)
        
    def forward(self, x_in):

        x_enc = [x_in]
        count = 0
        
        for i in range(self.depth):
            xenc = self.down(x_enc[-1])
            for _ in range(self.conv_repeats[count]):
                if i == 0:
                    x = self.in_conv(xenc)
                else:
                    x = self.super_block(xenc, self.W, self.b)
            x_enc.append(x)
            count+=1

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(self.depth):
            y = self.upsample(y)
            for _ in range(self.conv_repeats[count]):
                if i == range(self.depth)[-1]:
                    out = self.out_conv(y)
                else:
                    y = self.super_block(y, self.W, self.b)
            count+=1
    
        return out