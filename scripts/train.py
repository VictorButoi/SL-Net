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
from superlayer.models import SUnet
from superlayer.utils import BrainD, dice_coeff, one_hot
from validate import eval_net

dir_img = '/home/gid-dalcaav/projects/neuron/data/t1_mix/proc/resize256-crop_x32-slice100/train/vols/'
dir_mask = '/home/gid-dalcaav/projects/neuron/data/t1_mix/proc/resize256-crop_x32-slice100/train/asegs/'
dir_checkpoint_1 = 'checkpoints_1/'
dir_checkpoint_2 = 'checkpoints_2/'

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=10_000)


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              dice=True,
              checkpoint=0,
              target_label_numbers=None,
              writer=None,
              train_path=None,
              val_path=None):
    
    train = BrainD(dir_img, dir_mask, id_file=train_path, label_numbers=target_label_numbers)
    val = BrainD(dir_img, dir_mask, id_file=val_path, label_numbers=target_label_numbers)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)            
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    n_val = len(val)
    n_train = len(train)
    dataset_size = n_val + n_train
    
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    ''')

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    criterion = dice_coeff
    
    train_scores = []
    val_scores = []
    
    train_vars = []
    val_vars = []
    
    train_soft = []
    
    sub_epoch_interval = (dataset_size // (10 * batch_size))
    
    running_train_soft = []
    running_train_losses = []

    for epoch in range(epochs):

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                net.train()

                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                one_hot_true_masks = one_hot(true_masks, net.n_classes)

         
                pred = net(imgs)
                
                comp = one_hot_true_masks
                
                loss = criterion(pred, comp)
                running_train_soft.append(loss.item())
                
                pred = torch.argmax(pred, axis=1).unsqueeze(1)
                hard_loss = criterion(pred, true_masks)
                    
                running_train_losses.append(hard_loss.item())

                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % 4 == 0:

                    val_score, val_var = eval_net(net, val_loader, device)

                    train_scores.append(np.average(running_train_losses))
                    val_scores.append(val_score)
                    train_vars.append(np.var(running_train_losses))
                    val_vars.append(val_var)
                    
                    train_soft.append(np.average(running_train_soft))
                       
                    running_train_soft = []
                    running_train_loss = []

                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info('Validation Dice Loss: {}'.format(val_score))
                    writer.add_scalar('Loss/test', val_score, global_step)

    writer.close()

    return train_scores, val_scores, train_vars, val_vars, train_soft


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
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()
