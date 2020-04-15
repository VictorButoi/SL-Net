import argparse
import logging
import os
import sys

import numpy as np
#from matplotlib import pyplot
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from unet import TiedUNet
from dice_loss import dice_coeff

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BrainD
from torch.utils.data import DataLoader, random_split

dir_checkpoint = 'checkpoints/'

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.3,
              save_cp=True,
              img_scale=0.5):
    
    
    dataset = BrainD('data/imgs/', 'data/masks/', max_images=50, scale=img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
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
    criterion = nn.BCEWithLogitsLoss()

    train_scores = []
    val_scores = []

    sub_epoch_interval = (len(dataset) // (10 * batch_size))

    running_train_loss = 0
    running_train_losses = []

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:

            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                pred = torch.sigmoid(masks_pred)
                pred = (pred > 0.5).float()
                pred_dice = dice_coeff(pred, true_masks)
                running_train_loss += pred_dice.item()
                running_train_losses.append(pred_dice.item())

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
                    
                    val_score = eval_net(net, val_loader, device)

                    train_scores.append(running_train_loss/sub_epoch_interval)
                    val_scores.append(val_score)

                    running_train_loss = 0
                    running_train_losses = []
                    
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    writer.add_images('masks/true', true_masks, global_step)
                    writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

    return train_scores, val_scores


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
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

    #net = UNet(n_channels=1, n_classes=1, bilinear=True)
    plot = False
    net = TiedUNet(n_channels=1, n_classes=1, bilinear=True)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        nRuns = 1
        overall_train_statistics = []
        overall_eval_statistics = []

        train_mean_scores = []
        train_variances = []
        val_mean_scores = []
        val_variances = []

        for i in range(nRuns):
            train_scores, val_scores = train_net(net=net,
                                                epochs=args.epochs,
                                                batch_size=args.batchsize,
                                                lr=args.lr,
                                                device=device,
                                                img_scale=args.scale,
                                                val_percent=args.val / 100)
            overall_train_statistics.append(train_scores)
            overall_eval_statistics.append(val_scores)
        
        if plot:
            """
            nRecords = len(overall_train_statistics[0])

            for i in range(nRecords):
                train_one_record = []
                val_one_record = []
                running_train = 0
                running_val = 0

                for j in range(nRuns):
                    eval_stat = overall_eval_statistics[j][i]
                    train_stat = overall_train_statistics[j][i]
                    running_val +=eval_stat
                    running_train += train_stat
                    val_one_record.append(eval_stat)
                    train_one_record.append(train_stat)

                train_mean_scores.append(running_train/nRuns)
                val_mean_scores.append(running_val/nRuns)
                train_variances.append(np.std(train_one_record))
                val_variances.append(np.std(val_one_record))
            

            eval_high_var = [a + b for a, b in zip(val_mean_scores, val_variances)]
            eval_low_var = [a - b for a, b in zip(val_mean_scores, val_variances)]

            train_high_var = [a + b for a, b in zip(train_mean_scores, train_variances)]
            train_low_var = [a - b for a, b in zip(train_mean_scores, train_variances)]
            
            pyplot.title('Dice Loss v. Epoch Training Set over 10 runs')
            pyplot.xlabel('Iteration Checkpoint')
            pyplot.ylabel('Dice')

            domain = range(len(val_mean_scores))
            pyplot.plot(domain, val_mean_scores, label="Eval Dice", color="blue")
            pyplot.fill_between(domain, eval_high_var, y2=eval_low_var, color="skyblue", alpha=0.5)

            domain = range(len(train_mean_scores))
            pyplot.plot(domain, train_mean_scores, label="Training Dice", color="orange")
            pyplot.fill_between(domain, train_high_var, y2=train_low_var, color="navajowhite", alpha=0.5)

            pyplot.legend()
            pyplot.savefig('plots/total_loss_' + str(nRuns) + '.png')
            pyplot.close()
            """
    
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)