import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

sys.path.append("..")
from superlayer.models import UNet
from superlayer.utils import BrainD, soft_dice, hard_dice, one_hot
from validate import eval_net
from matplotlib import pyplot as plt

dir_img = '/nfs02/users/gid-dalcaav/projects/neuron/data/t1_mix/proc/resize256-crop_x32-slice100/train/vols/'
dir_mask = '/nfs02/users/gid-dalcaav/projects/neuron/data/t1_mix/proc/resize256-crop_x32-slice100/train/asegs/'

def train_net(net,
              epochs,
              batch_size,
              lr,
              target_label_numbers,
              train_path=None,
              val_path=None,
              segment=True):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    
    train = BrainD(dir_img, dir_mask, id_file=train_path, label_numbers=target_label_numbers)
    val = BrainD(dir_img, dir_mask, id_file=val_path, label_numbers=target_label_numbers)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)            
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    
    global_step = 0
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    if segment:
        criterion = soft_dice
    else:
        criterion = nn.MSELoss()
    
    train_soft, val_scores = [], []
    train_vars, val_vars = [], []
    
    sub_epoch_interval = (len(train) // (4 * batch_size))
    
    running_train_soft = []

    for epoch in range(epochs):

        with tqdm(total=len(train), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                net.train()

                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                one_hot_true_masks = one_hot(true_masks, net.n_classes)
         
                pred = net(imgs)
                comp = one_hot_true_masks if segment else imgs
                
                loss = criterion(pred, comp)
                running_train_soft.append(loss.item())

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % sub_epoch_interval == 0:

                    val_score, val_var = eval_net(net, val_loader, device, segment)

                    train_soft.append(np.average(running_train_soft))
                    train_vars.append(np.var(running_train_soft))
                    val_scores.append(val_score)
                    val_vars.append(val_var)
                    
                       
                    running_train_soft = []
                    
                    logging.info('Validation Dice Loss: {}'.format(val_score))

    return train_soft, val_scores, train_vars, val_vars


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=3,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()
