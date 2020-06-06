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
from superlayer.models import BlockLearner, SuperNet
from superlayer.utils import BrainD, dice_coeff, one_hot
from validate import eval_net

dir_img = '/home/gid-dalcaav/projects/neuron/data/t1_mix/proc/resize256-crop_x32-slice100/train/vols/'
dir_mask = '/home/gid-dalcaav/projects/neuron/data/t1_mix/proc/resize256-crop_x32-slice100/train/asegs/'
dir_checkpoint_1 = 'checkpoints_1/'
dir_checkpoint_2 = 'checkpoints_2/'

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=10_000)


def train_block(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1,
              dice=True,
              checkpoint=0,
              target_label_numbers=None,
              dataset=None,
              train_loader=None,
              val_loader=None,
              writer=None,
              jupyterN=True,
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
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    
    enc_nf = [128, 128, 128, 128]
    dec_nf = [128, 128, 128, 128]
    bl_net = BlockLearner(input_ch=256, out_ch=256, use_bn=True, enc_nf=enc_nf, dec_nf=dec_nf, super_block_dim=[128,256,3,3])
    net.eval()

    optimizer = optim.RMSprop(bl_net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    criterion = dice_coeff
    
    train_scores = []
    val_scores = []
    
    train_vars = []
    val_vars = []

    sub_epoch_interval = (dataset_size // (10 * batch_size))

    running_train_losses = []

    for epoch in range(epochs):

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                bl_net.train()

                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                one_hot_true_masks = one_hot(true_masks, net.n_classes)
                
                outweight = bl_net()
                masks_pred = net(imgs, outweight)

                loss = criterion(masks_pred, one_hot_true_masks)
                epoch_loss += loss.item()

                pred = torch.argmax(masks_pred, axis=1).unsqueeze(1)
                hard_loss = criterion(pred, true_masks)
                running_train_losses.append(hard_loss.item())

                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % sub_epoch_interval == 0:

                    val_score, val_var = eval_net(net, val_loader, device, weight=outweight)

                    train_scores.append(np.average(running_train_losses))
                    val_scores.append(val_score)
                    train_vars.append(np.var(running_train_losses))
                    val_vars.append(val_var)

                    running_train_loss = []

                    scheduler.step(val_score)

                    logging.info('Validation Dice Loss: {}'.format(val_score))
                    writer.add_scalar('Loss/test', val_score, global_step)

    writer.close()

    return train_scores, val_scores, train_vars, val_vars
