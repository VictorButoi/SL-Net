import sys
import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .unet_parts import simple_block
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class BlockLearner(nn.Module):
    
    def __init__(self, input_ch, out_ch, use_bn, superblock_size, depth, W=None):
        super(BlockLearner, self).__init__()
        

        
    def forward(self, x_in):
        
        
    
