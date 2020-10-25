import sys
import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .unet_parts import simple_block
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class UNet(nn.Module):
    def __init__(self, input_ch, out_ch, enc_nf, dec_nf):
        super(UNet, self).__init__()
        
        self.n_classes = out_ch
        self.down = torch.nn.MaxPool2d(2,2)

        self.block0 = simple_block(input_ch , enc_nf[0], True)
        
        self.block1 = simple_block(enc_nf[0], enc_nf[1], True)
        self.block2 = simple_block(enc_nf[1], enc_nf[2], True)
        self.block3 = simple_block(enc_nf[2], enc_nf[3], True)
        self.block4 = simple_block(enc_nf[3], dec_nf[0], True)    
        
        self.block5 = simple_block(dec_nf[0]*2, dec_nf[1], True)       
        self.block6 = simple_block(dec_nf[1]*2, dec_nf[2], True)         
        self.block7 = simple_block(dec_nf[2]*2, dec_nf[3], True) 
        self.block8 = simple_block(dec_nf[3]*2, dec_nf[3], True)           

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
