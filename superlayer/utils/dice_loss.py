import torch
from torch.autograd import Function
import numpy as np

def dice_coeff(pred, target):
    
    smooth = 1.

    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    dice_sim = ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
    dice_loss = 1 - dice_sim
    
    return dice_loss

def one_hot(targets, C):    
    targets_extend=targets.clone().long()
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1) 
    return one_hot