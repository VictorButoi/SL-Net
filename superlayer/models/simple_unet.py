import sys
import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class SUnet(nn.Module):
    def __init__(self, input_ch, use_bn, enc_nf, dec_nf):
        super(SUnet, self).__init__()
        
        self.block0 = conv_block(input_ch, enc_nf[0], 2, use_bn)
        self.block1 = conv_block(enc_nf[0], enc_nf[1], 2, use_bn)
        self.block2 = conv_block(enc_nf[1], enc_nf[2], 2, use_bn)
        self.block3 = conv_block(enc_nf[2], enc_nf[3], 2, use_bn)

        self.block4 = conv_block(enc_nf[3], dec_nf[0], 1, use_bn)           #1
        self.block5 = conv_block(dec_nf[0]*2, dec_nf[1], 1, use_bn)         #2
        self.block6 = conv_block(dec_nf[1]*2, dec_nf[2], 1, use_bn)         #3
        self.block7 = conv_block(dec_nf[2]+enc_nf[0], dec_nf[3], 1, use_bn) #4
        self.block8 = conv_block(dec_nf[3], dec_nf[4], 1, use_bn)           #5
        self.block9 = conv_block(dec_nf[4]+input_ch, dec_nf[5], 1, use_bn)

        self.out = nn.Conv3d(dec_nf[5], dec_nf[6], kernel_size=3, padding=1)


    def forward(self, x_in):
        #Model
        x0 = self.block0(x_in)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        x = self.block4(x3)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        if not x.size() == x2.size():
            x = x[:,:,1:,:,1:]
            
        x = torch.cat([x, x2], 1)
        x = self.block5(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
 
        x = torch.cat([x, x1], 1)
        x = self.block6(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        x = torch.cat([x, x0], 1)
        x = self.block7(x)
        x = self.block8(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = torch.cat([x, x_in], 1)
        x = self.block9(x)
        
        out = self.out(x)
            
        return out

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_bn):
        super(conv_block, self).__init__()
        
        self.use_bn= use_bn
        
        if stride == 1:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride= stride, padding=1)
            self.bn = nn.InstanceNorm3d(out_channels)
            self.activation = nn.LeakyReLU(0.2)

        elif stride == 2:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride= stride, padding=1)
            self.bn = nn.InstanceNorm3d(out_channels)
            self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)
        out = self.activation(out)
        return out