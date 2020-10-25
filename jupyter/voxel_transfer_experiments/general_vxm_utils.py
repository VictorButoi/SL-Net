import sys
import os

sys.path.append("/home/vib9/src/SL-Net/superlayer/voxel")

from train import train
import argparse
import logging
import os
import sys

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

sys.path.append("../../")
sys.path.append("../../scripts/")
import superlayer.utils

from scripts import train_net, get_args

from superlayer.models import SLN_AE, AE, cvpr2018_net, UNet, SLN_UNet, sln_cvpr2018_net
from superlayer.utils import BrainD, hard_dice, soft_dice, one_hot, plot_img_array, plot_side_by_side

sys.path.append("/home/vib9/src/voxelmorph/pytorch")
import datagenerators
from models import SpatialTransformer

sys.path.append("/home/vib9/src/voxelmorph/ext/medipy-lib/medipy")
from metrics import dice

dir_img = '/nfs02/users/gid-dalcaav/projects/neuron/data/t1_mix/proc/resize256-crop_x32-slice100/train/vols/'
dir_mask = '/nfs02/users/gid-dalcaav/projects/neuron/data/t1_mix/proc/resize256-crop_x32-slice100/train/asegs/'

dir_checkpoint_1 = 'checkpoints_1/'
dir_checkpoint_2 = 'checkpoints_2/'

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_file = '/home/vib9/src/SL-Net/jupyter/partitions/train.txt'
val_file = '/home/vib9/src/SL-Net/jupyter/partitions/val.txt'
atlas_file = '/home/vib9/src/voxelmorph/data/atlas_norm.npz'

atlas_vol = np.load(atlas_file)['vol'][np.newaxis, ..., np.newaxis][:,:,:,100,:]
vol_size = atlas_vol.shape[1:-1]

def save_results(model, train_data, val_data, train_std, val_std, save_name, slblock=False):
    root = "/home/vib9/src/SL-Net/jupyter/voxel_transfer_experiments/"
    torch.save(model.state_dict(), root + "final_models/" + save_name)
    
    train_dice = np.asarray(train_data)[np.newaxis,:]
    val_dice = np.asarray(val_data)[np.newaxis,:]
    
    train_std = np.asarray(train_std)[np.newaxis,:]
    val_std = np.asarray(val_std)[np.newaxis,:]
    
    dice_scores = np.concatenate((train_dice,val_dice,train_std,val_std),axis=0)
    np.save(root + "/results/" + save_name, dice_scores)
    
def plot_subplot_array(loss_array1, loss_array2, axis_lims, titles):
    root = "/home/vib9/src/SL-Net/jupyter/voxel_transfer_experiments/results"
    
    loss_array1 = np.load(root + "/" + loss_array1 + ".npy")
    loss_array2 = np.load(root + "/" + loss_array2 + ".npy")
    
    train_dice1 = loss_array1[0]
    val_dice1 = loss_array1[1]
    train_std1 = loss_array1[2]
    val_std1 = loss_array1[3]
    
    train_dice2 = loss_array2[0]
    val_dice2 = loss_array2[1]
    train_std2 = loss_array2[2]
    val_std2 = loss_array2[3]
    
    if len(loss_array1) > 4:
        train_soft1 = loss_array1[4]
        train_soft2 = loss_array2[4]
    
    ziptrainup1 = [a + b for a, b in zip(train_dice1, train_std1)]
    ziptraindown1 = [a - b for a, b in zip(train_dice1, train_std1)]
    zipvalup1 = [a + b for a, b in zip(val_dice1, val_std1)]
    zipvaldown1 = [a - b for a, b in zip(val_dice1, val_std1)]
    
    ziptrainup2 = [a + b for a, b in zip(train_dice2, train_std2)]
    ziptraindown2 = [a - b for a, b in zip(train_dice2, train_std2)]
    zipvalup2 = [a + b for a, b in zip(val_dice2, val_std2)]
    zipvaldown2 = [a - b for a, b in zip(val_dice2, val_std2)]
    
    domain = len(train_dice1)
    x_values = [i for i in range(domain)]
    
    f = plt.figure(figsize=(15,5))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    
    ax.set_title(titles[0])
    ax.set_ylim(axis_lims)
    
    ax.plot(x_values, train_dice1, color="blue", label="train hard dice")
    ax.fill_between(x_values, ziptrainup1, ziptraindown1, facecolor='lightskyblue', alpha=0.5)
    ax.plot(x_values, val_dice1, color="orange", label="val hard dice")
    ax.fill_between(x_values, zipvalup1, zipvaldown1, facecolor='navajowhite', alpha=0.5)
    
    if len(loss_array1) > 4:
        ax.plot(x_values, train_soft1, color="red", label="train soft dice")

    ax.set_xlabel("Mini-epochs")
    ax.set_ylabel("Loss Val")
    ax.legend()
    ax.grid()
    
    ax2.set_title(titles[1])
    ax2.set_ylim(axis_lims)
    
    ax2.plot(x_values, train_dice2, color="blue", label="train hard dice")
    ax2.fill_between(x_values, ziptrainup2, ziptraindown2, facecolor='lightskyblue', alpha=0.5)
    ax2.plot(x_values, val_dice2, color="orange", label="val hard dice")
    ax2.fill_between(x_values, zipvalup2, zipvaldown2, facecolor='navajowhite', alpha=0.5)
    
    if len(loss_array1) > 4:
        ax2.plot(x_values, train_soft2, color="red", label="train soft dice")
        
    ax2.set_xlabel("Mini-epochs")
    ax2.set_ylabel("Loss Val")
    ax2.legend()
    ax2.grid()

    plt.show()


def plot_image_results(model, mod_name, data, model_type="vxm"):
    model.to(device)
    model.load_state_dict(torch.load("/home/vib9/src/SL-Net/jupyter/voxel_transfer_experiments/final_models/" + mod_name))
    model.eval()
    
    if model_type=="vxm":
        atlas_file = '/home/vib9/src/voxelmorph/data/atlas_norm.npz'
        atlas = np.load(atlas_file)
        atlas_image = atlas['vol'][:,:,100]
        atlas_seg = atlas['seg'][:,:,100]

        atlas_vol = atlas['vol'][np.newaxis, ..., np.newaxis]
        input_fixed  = torch.from_numpy(atlas_vol).to(device).float()[:,:,:,100,:]
        input_fixed  = input_fixed.permute(0, 3, 1, 2)

        atlas_slice = atlas['vol'][np.newaxis, ..., np.newaxis][:,:,:,100,:]
        trf = SpatialTransformer(atlas_slice.shape[1:-1], mode='nearest')
        trf.to(device)

        string_file = open(train_file) if data=="train" else open(val_file)
        strings = string_file.readlines()
        vol_names = [dir_i + x.strip() + ".npz" for x in strings]
        example_gen = datagenerators.example_gen(vol_names, 1, return_segs=True)

        moving_image, moving_seg = next(example_gen)
        moving_image = torch.from_numpy(moving_image).to(device).float().permute(0, 3, 1, 2)

        warp, flow = model(moving_image, input_fixed)

        # Warp segment using flow
        moving_seg = torch.from_numpy(moving_seg).to(device).float()
        moving_seg = moving_seg.permute(0, 3, 1, 2)
        warp_seg = trf(moving_seg, flow).detach().cpu().numpy()
        warp_img = trf(moving_image, flow).detach().cpu().numpy()

        dice_score = 1 - np.average(dice(warp_seg, atlas_seg,labels=target_label_numbers))

        display_image = moving_image.squeeze(0).squeeze(0).cpu()
        display_seg = moving_seg.squeeze(0).squeeze(0).cpu()
        display_warp_seg = warp_seg.squeeze(0).squeeze(0)
        display_warp_img = warp_img.squeeze(0).squeeze(0)
        
        unique_seg = np.unique(display_seg)
        unique_atlas = np.unique(atlas_seg)
        combined = np.setdiff1d(unique_seg,unique_atlas)

        for label in combined:
            display_image = np.where(display_image == label, 0, display_image)
            display_seg = np.where(display_seg == label, 0, display_seg)
            display_warp_seg = np.where(display_warp_seg == label, 0, display_warp_seg)
            display_warp_img = np.where(display_warp_img == label, 0, display_warp_img)
            atlas_image = np.where(atlas_image == label, 0, atlas_image)
            atlas_seg = np.where(atlas_seg == label, 0, atlas_seg)
        
        f, axarr = plt.subplots(2,3,figsize=(15,8))
        
        axarr[0,0].set_title("Image")
        axarr[0,0].imshow(display_image)

        axarr[0,1].set_title("Atlas Image")
        axarr[0,1].imshow(atlas_image)

        axarr[0,2].set_title("Predicted Image")
        axarr[0,2].imshow(display_warp_img)

        axarr[1,0].set_title("Segmentation")
        axarr[1,0].imshow(display_seg)

        axarr[1,1].set_title("Atlas Segmentation")
        axarr[1,1].imshow(atlas_seg)

        axarr[1,2].set_title("Predicted Segmentation")
        axarr[1,2].imshow(display_warp_seg)

        
    elif model_type=="unet":
            
        dataset = BrainD(dir_img, dir_mask, id_file=data, label_numbers=target_label_numbers)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
        criterion = dice_coeff
        
        example = next(iter(loader))
        
        imgs = example['image']
        true_masks = example['mask']

        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)
        one_hot_true_masks = one_hot(true_masks, model.n_classes)

        pred = model(imgs)
        comp = one_hot_true_masks

        loss = criterion(pred, comp)

        pred = torch.argmax(pred, axis=1).unsqueeze(1)
        dice_score = criterion(pred, true_masks)
        
        imgs = imgs.squeeze(0).squeeze(0).cpu().data
        true_masks = true_masks.squeeze(0).squeeze(0).cpu().data
        pred = pred.squeeze(0).squeeze(0).cpu().data
        
        f, axarr = plt.subplots(1,3,figsize=(15,8))
        
        axarr[0].set_title("Image")
        axarr[0].imshow(imgs)

        axarr[1].set_title("Segmentation")
        axarr[1].imshow(true_masks)

        axarr[2].set_title("Prediction")
        axarr[2].imshow(pred)
    
    else:
        raise ValueError("Not implemented")
    
    print("Loss: " + str(dice_score))