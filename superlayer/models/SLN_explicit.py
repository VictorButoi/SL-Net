import sys
import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .unet_parts import simple_block
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class SuperNet(nn.Module):
    
    def __init__(self, input_ch, out_ch, use_bn, superblock_size, depth, W=None, firstW=None, lastW=None, learnB=False):
        super(SuperNet, self).__init__()
        
        self.superblock_size = superblock_size
        self.depth = depth
        self.n_classes = out_ch
        self.down = torch.nn.MaxPool2d(2,2)
        half_size = int(superblock_size/2)
        self.use_bn = use_bn
        hW = None
        
        #Kernel size is 3
        if W==None:
            if not learnB:
                self.W = torch.nn.Parameter(torch.randn(half_size, superblock_size,3,3))
                self.W.requires_grad = True
                hW = self.W[:,:half_size,:,:]
                hW.requires_grad = True
            else:
                self.W = torch.zeros(half_size, superblock_size,3,3)
        else:
            self.W = W
            if not torch.is_tensor(self.W):
                self.W = torch.from_numpy(self.W)
        

        self.block0 = simple_block(input_ch , half_size, use_bn, weight=firstW)   
        self.down_block = simple_block(half_size, half_size, use_bn, weight=hW)
        self.up_block = simple_block(superblock_size, half_size, use_bn, weight=self.W)
        
        if lastW is None:
            self.out_conv = nn.Conv2d(half_size, out_ch, kernel_size=3, padding=1)
        else:
            self.out_conv = F.conv2d(x, lastW, padding=1)
            
        self.sm = nn.Softmax(dim=1)

        
    def forward(self, x_in, learned_weight=None):
        
        if not learned_weight is None:
            learned_weight = learned_weight.detach()
            
            half_size = int(self.superblock_size/2)
            half_learned_weight = learned_weight[:,:half_size,:,:]
            self.down_block = simple_block(half_size, half_size, self.use_bn, weight=half_learned_weight)
            self.up_block = simple_block(self.superblock_size, half_size, self.use_bn, weight=learned_weight)
            
        #Model
        
        x = self.block0(x_in)
        
        enc_seq = [x]
        
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