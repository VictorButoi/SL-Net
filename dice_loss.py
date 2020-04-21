import torch
from torch.autograd import Function

def dice_coeff(pred, target):
    eps = 1
    
    assert pred.size() == target.size(), 'Input and target are different dim'
    if len(target.size())==4:
        n,c,x,y = target.size()
    if len(target.size())==5:
        n,c,x,y,z = target.size()
    target = target.view(n,c,-1)
    pred = pred.view(n,c,-1)

    num = torch.sum(2*(target*pred),2) + eps
    den = (pred*pred).sum(2) + (target*target).sum(2) + eps
    dice_loss = 1-num/den
    ind_avg = dice_loss
    total_avg = torch.mean(dice_loss.float())
    regions_avg = torch.mean(dice_loss.float(), 0)
    return total_avg, regions_avg, ind_avg

def one_hot(targets, C):    
    targets_extend=targets.clone()
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1) 
    return one_hot