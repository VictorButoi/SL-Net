import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import sys

sys.path.append("..")
from superlayer.utils import BrainD, dice_coeff, one_hot


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch

    running_hard_loss = []

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred = net(imgs)

            pred = torch.argmax(pred, axis=1).unsqueeze(1)
            loss = dice_coeff(pred, true_masks).item()
            
            running_hard_loss.append(loss)

            pbar.update()

    return np.average(running_hard_loss), np.var(running_hard_loss) 
