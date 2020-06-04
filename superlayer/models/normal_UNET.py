import sys
import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class SUnet(nn.Module):
    def __init__(self, input_ch, out_ch, use_bn, enc_nf, dec_nf, ignore_last=False, W=None):
        super(SUnet, self).__init__()
        
        hW = W:
        if not W==None:
            second_dim = W.shape[1]
            hW = W[:,:int(second_dim/2),:,:]
            
        
        self.n_classes = out_ch
        self.ignore_last = ignore_last
        self.down = torch.nn.MaxPool2d(2,2)

        self.block0 = simple_block(input_ch , enc_nf[0], use_bn)
        
        
        self.block1 = simple_block(enc_nf[0], enc_nf[1], use_bn, weight=hW)
        self.block2 = simple_block(enc_nf[1], enc_nf[2], use_bn, weight=hW)
        self.block3 = simple_block(enc_nf[2], enc_nf[3], use_bn, weight=hW)
        self.block4 = simple_block(enc_nf[3], dec_nf[0], use_bn, weight=hW)    
        
        self.block5 = simple_block(dec_nf[0]*2, dec_nf[1], use_bn, weight=W)       
        self.block6 = simple_block(dec_nf[1]*2, dec_nf[2], use_bn, weight=W)         
        self.block7 = simple_block(dec_nf[2]*2, dec_nf[3], use_bn, weight=W) 
        self.block8 = simple_block(dec_nf[3]*2, dec_nf[3], use_bn, weight=W)           

        self.out_conv = nn.Conv2d(dec_nf[3], out_ch, kernel_size=3, padding=1)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x_in):

        #Model
        x0 = self.block0(x_in)
        
        x1 = self.block1(self.down(x0))
        
        x2 = self.block2(self.down(x1))
        
        x3 = self.block3(self.down(x2))
        
        x = self.block4(self.down(x3))
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        x = torch.cat([x, x3], 1)
        
        x = self.block5(x)
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
 
        x = torch.cat([x, x2], 1)
        x = self.block6(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        x = torch.cat([x, x1], 1)
        x = self.block7(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = torch.cat([x, x0], 1)
        x = self.block8(x)
        
        out = self.out_conv(x)
        out = self.sm(out)
        
        return out

class simple_block(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(simple_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.use_bn= use_bn
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.InstanceNorm2d(out_channels)  
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.activation(out)
        return out