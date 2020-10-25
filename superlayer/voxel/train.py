"""
*Preliminary* pytorch implementation.

VoxelMorph training.
"""
#test comment
from collections import Counter

import sys
sys.path.append("/home/vib9/src/voxelmorph-sandbox/pytorch/")
sys.path.append("/home/vib9/src/SL-Net/superlayer/")
sys.path.append("/home/vib9/src/voxelmorph/ext/medipy-lib/medipy")

from metrics import dice

# python imports
import os
import glob
import random
import warnings
from argparse import ArgumentParser

# external imports
import numpy as np
import torch
from torch.optim import Adam

import datagenerators
import losses
import matplotlib.pyplot as plt

from models import SpatialTransformer

sys.path.append("../voxel/")


import scipy.io as sio


def train(model,
          data_dir,
          train_file,
          val_file,
          atlas_file,
          n_iter,
          batch_size,
          mod_name,
          gpu='0',
          lr=1e-4,
          data_loss='mse',
          reg_param=0.01,
          train_module=None,
          target_label_numbers=None):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    # Produce the loaded atlas with dims.:160x192x224.
    atlas = np.load(atlas_file)
    atlas_vol = atlas['vol'][np.newaxis, ..., np.newaxis]
    atlas_seg = atlas['seg'][:,:,100]

    train_file = open(train_file)
    train_strings = train_file.readlines()
    train_vol_names = [data_dir + x.strip() + ".npz" for x in train_strings]
        
    model.to(device)

    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)

    sim_loss_fn = losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
    train_example_gen = datagenerators.example_gen(train_vol_names, batch_size, return_segs=True)

    # set up atlas tensor
    atlas_vol_bs = np.repeat(atlas_vol, batch_size, axis=0)
    input_fixed  = torch.from_numpy(atlas_vol_bs).to(device).float()[:,:,:,100,:]
    
    input_fixed  = input_fixed.permute(0, 3, 1, 2)
    
    train_std = []
    val_std = []
    
    train_dices = []
    val_dices = []
    
    train_dice_acc = []
    # Use this to warp segments
    
    atlas_slice = atlas['vol'][np.newaxis, ..., np.newaxis][:,:,:,100,:]
    
    trf = SpatialTransformer(atlas_slice.shape[1:-1], mode='nearest')
    trf.to(device)
    
    for i in range(n_iter):
     
        model.train()

        # Generate the moving images and convert them to tensors.
        moving_image, moving_seg = next(train_example_gen)
        
        input_moving = torch.from_numpy(moving_image).to(device).float()
        input_moving = input_moving.permute(0, 3, 1, 2)
        warp, flow = model(input_moving, input_fixed)
        
        # Warp segment using flow
        moving_seg = torch.from_numpy(moving_seg).to(device).float()
        moving_seg = moving_seg.permute(0, 3, 1, 2)
        warp_seg = trf(moving_seg, flow).detach().cpu().squeeze(0).squeeze(0).numpy()
        
        dice_score = np.average(dice(warp_seg, atlas_seg, labels=target_label_numbers))
        train_dice_acc.append(dice_score)          
        
        # Calculate loss
        recon_loss = sim_loss_fn(warp, input_fixed) 
        grad_loss = grad_loss_fn(flow)
        loss = recon_loss + reg_param * grad_loss

        print("Train Epoch: %d | Loss: %f | Reconstruction Loss: %f | Dice Score: %f"\
                 % (i, loss.item(), recon_loss.item(), dice_score.item()), flush=True)

        # Backwards and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 1000 == 0:
            
            test_dices = test_net(gpu, model, atlas_file, val_file, data_dir, target_label_numbers)
            
            train_dices.append(np.average(train_dice_acc))
            val_dices.append(np.average(test_dices))
            
            train_std.append(np.std(train_dice_acc))
            val_std.append(np.std(test_dices))
            
            train_dice_acc = []
            
    return train_dices, val_dices, train_std, val_std


def test_net(gpu, 
             model,
             atlas_file,
             val_file,
             data_dir,
             target_label_numbers):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param init_model_file: the model directory to load from
    """
    model.eval()

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
   
    # Produce the loaded atlas with dims.:160x192x224.
    atlas = np.load(atlas_file)
    atlas_vol = atlas['vol'][np.newaxis, ..., np.newaxis][:,:,:,100,:]
    atlas_seg = atlas['seg'][:,:,100]
    vol_size = atlas_vol.shape[1:-1]

    val_file = open(val_file)
    val_strings = val_file.readlines()
    
    def create_vol_name(data_dir, x):
        vol_string = data_dir + x + ".npz"
        aseg_string = vol_string.replace("vols", "asegs").replace("norm","aseg")
        return vol_string + "," + aseg_string
        
    val_vol_names = [create_vol_name(data_dir, x.strip()) for x in val_strings]

    # set up atlas tensor
    input_fixed  = torch.from_numpy(atlas_vol).to(device).float()
    input_fixed  = input_fixed.permute(0, 3, 1, 2)

    # Use this to warp segments
    
    trf = SpatialTransformer(vol_size, mode='nearest')
    trf.to(device)
    
    total_dice = []

    for k in range(0, len(val_vol_names)):

        vol_name, seg_name = val_vol_names[k].split(",")
        
        X_vol, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)

        input_moving  = torch.from_numpy(X_vol).to(device).float()
        input_moving  = input_moving.permute(0, 3, 1, 2)
        
        warp, flow = model(input_moving, input_fixed)

        # Warp segment using flow
        moving_seg = torch.from_numpy(X_seg).to(device).float()
        moving_seg = moving_seg.permute(0, 3, 1, 2)
        
        warp_seg = trf(moving_seg, flow).detach().cpu().squeeze(0).squeeze(0).numpy()

        dice_score = np.average(dice(warp_seg, atlas_seg,labels=target_label_numbers))
        print("Val iter " + str(k) + ": %f" % (dice_score), flush=True)

        total_dice.append(dice_score)
        
    return total_dice 
        
        