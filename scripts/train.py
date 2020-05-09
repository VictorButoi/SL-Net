import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
sys.path.append("../superlayer/models/")
from models import UNet
from models import TiedUNet

from torch.utils.tensorboard import SummaryWriter
sys.path.append("../superlayer/utils/")
from dataset import BrainD
from torch.utils.data import DataLoader, random_split

from dice_loss import dice_coeff
from dice_loss import one_hot

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
              writer=None):

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    
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
        if dice:
            criterion = dice_coeff
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    train_scores = []
    val_scores = []
    
    train_vars = []
    val_vars = []
    

    sub_epoch_interval = (len(dataset) // (10 * batch_size))

    running_train_loss = 0

    for epoch in range(epochs):

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                net.train()

                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                one_hot_true_masks = one_hot(true_masks, net.n_classes)

                masks_pred = net(imgs)

                loss = criterion(masks_pred, one_hot_true_masks)
                
                
                running_train_loss += loss.item()
                epoch_loss += loss.item()

                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % sub_epoch_interval == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    val_score, val_var = eval_net(net, val_loader, device)

                    train_scores.append(running_train_loss/sub_epoch_interval)
                    val_scores.append(val_score)
                    running_train_loss = 0

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

    #   - For N > 2 classes, use n_classes=N
    net1 = UNet(n_channels=1, n_classes=15, bilinear=True)
    net2 = TiedUNet(n_channels=1, n_classes=15, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    net1.to(device=device)
    net2.to(device=device)
    # faster convolutions, but more memory

    train_scores, val_scores = train_net(net=net1,
                                            epochs=args.epochs,
                                            batch_size=args.batchsize,
                                            lr=args.lr,
                                            device=device,
                                            img_scale=args.scale,
                                            val_percent=args.val / 100,
                                            checkpoint=1)
    
    train_scores, val_scores = train_net(net=net2,
                                            epochs=args.epochs,
                                            batch_size=args.batchsize,
                                            lr=args.lr,
                                            device=device,
                                            img_scale=args.scale,
                                            val_percent=args.val / 100,
                                            checkpoint=2)
