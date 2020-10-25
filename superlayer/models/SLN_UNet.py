import sys
import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function

from .unet_parts import simple_block
from .unet_parts import FeatureWeighter
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class SLN_UNet(nn.Module):
    
    def __init__(self, input_ch, out_ch, superblock_size, depth, W=None, b=None, conv_num=None, train_block=True, retrain=False):
        super(SLN_UNet, self).__init__()
            
        self.sb_halfsize = int(superblock_size/2)
        self.sb_size = superblock_size
        self.depth = depth
        self.n_classes = out_ch
        
        self.down = torch.nn.MaxPool2d(2,2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.in_conv = simple_block(in_channels=input_ch , out_channels=self.sb_halfsize)
        self.super_block = simple_block(weight=W, bias=b, sb_size=superblock_size, train_block=train_block)
        
        if retrain:
            self.fw_enc = nn.ModuleList()
            for i in range(depth - 1):
                self.enc.append(FeatureWeighter(W))
                
            self.fw_dec = nn.ModuleList()
            for i in range(depth - 1):
                self.dec.append(FeatureWeighter(W))
        
        self.out_conv = nn.Conv2d(self.sb_halfsize + 1, out_ch, kernel_size=3, padding=1)
        
        self.retrain = retrain
        self.sm = nn.Softmax(dim=1)

        
    def forward(self, x_in):
        
        x_enc = [x_in]
        
        for i in range(self.depth):
            xenc = self.down(x_enc[-1])
            if i == 0:
                x = self.in_conv(xenc)
            else:
                if self.retrain:
                    x = self.fw_enc[i](xenc)
                else:     
                    x = self.super_block(xenc, use_half=True)
            x_enc.append(x)

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(self.depth):
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i+2)]], dim=1)
            if i == range(self.depth)[-1]:
                out = self.out_conv(y)
            else:
                if self.retrain:
                    y = self.fw_dec[i](y)
                else:
                    y = self.super_block(y)
       
        out = self.sm(out)
        
        return out