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
              val_path=None,
              segment=True):
    
    if not jupyterN:
        target_label_numbers = [0,2,3,4,10,16,17,28,31,41,42,43,49,53,63]

        dataset = BrainD(dir_img, dir_mask, label_numbers=target_label_numbers)
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

        writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    
    if not(train_path == None or val_path == None):
        train = BrainD(dir_img, dir_mask, id_file=train_path, label_numbers=target_label_numbers)
        val = BrainD(dir_img, dir_mask, id_file=val_path, label_numbers=target_label_numbers)
        
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)            
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
        
        n_val = len(val)
        n_train = len(train)
        dataset_size = n_val + n_train
    else:
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
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

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

    if net.n_classes > 1:
        if segment:
            if dice:
                criterion = dice_coeff
            else:
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
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
                net.train()

                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                one_hot_true_masks = one_hot(true_masks, net.n_classes)

         
                pred = net(imgs)
                
                if segment:
                    comp = one_hot_true_masks
                else:
                    comp = imgs
                
                loss = criterion(pred, comp)
                epoch_loss += loss.item()
                
                if segment:
                    pred = torch.argmax(pred, axis=1).unsqueeze(1)
                    hard_loss = criterion(pred, true_masks)
                else:
                    hard_loss = loss
                    
                running_train_losses.append(hard_loss.item())

                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % sub_epoch_interval == 0:

                    val_score, val_var = eval_net(net, val_loader, device, segment)

                    train_scores.append(np.average(running_train_losses))
                    val_scores.append(val_score)
                    train_vars.append(np.var(running_train_losses))
                    val_vars.append(val_var)

                    running_train_loss = []

                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation Dice Loss: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

        if save_cp:
            if checkpoint == 1:
                try:
                    os.mkdir(dir_checkpoint_1)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),dir_checkpoint_1 + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')
            else:
                try:
                    os.mkdir(dir_checkpoint_2)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),dir_checkpoint_2 + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

    return train_scores, val_scores, train_vars, val_vars


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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    enc_nf = [4, 8, 16, 32]
    dec_nf = [32, 16, 8, 4]
    
    net = SLNet(input_ch=1, out_ch=15, use_bn=True, superblock_size=64, depth=4)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    net.to(device=device)

    train_scores1, val_scores1, train_vars1, val_vars1 = train_net(net=net,
                                            epochs=args.epochs,
                                            batch_size=args.batchsize,
                                            lr=args.lr,
                                            device=device,
                                            img_scale=args.scale,
                                            val_percent=args.val / 100,
                                            checkpoint=1,
                                            jupyterN=False)
