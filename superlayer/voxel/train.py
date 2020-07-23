"""
*Preliminary* pytorch implementation.

VoxelMorph training.
"""
#test comment
import sys
sys.path.append("/home/vib9/src/voxelmorph-sandbox/pytorch/")
sys.path.append("/home/vib9/src/SL-Net/superlayer/")

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

from models import SpatialTransformer

sys.path.append("../voxel/")


import scipy.io as sio

sys.path.append('/home/vib9/src/voxelmorph/ext/medipy-lib')
from medipy.metrics import dice


def train(mod,
          gpu,
          data_dir,
          train_file,
          val_file,
          atlas_file,
          lr,
          n_iter,
          data_loss,
          model,
          reg_param, 
          batch_size,
          train_module=None,
          target_label_numbers=None):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    # Produce the loaded atlas with dims.:160x192x224.
    atlas = np.load(atlas_file)
    atlas_vol = atlas['vol'][np.newaxis, ..., np.newaxis]
    atlas_seg = atlas['seg'][:,:,100]

    # Get all the names of the training data
    
    train_file = open(train_file)
    train_strings = train_file.readlines()
    train_vol_names = [data_dir + x.strip() + ".npz" for x in train_strings]
    
    model = mod
        
    model.to(device)

    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)

    sim_loss_fn = losses.ncc_loss if data_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
    train_example_gen = datagenerators.example_gen(train_vol_names, batch_size, return_segs=True)

    # set up atlas tensor
    atlas_vol_bs = np.repeat(atlas_vol, batch_size, axis=0)
    input_fixed  = torch.from_numpy(atlas_vol_bs).to(device).float()[:,:,:,100,:]
    
    input_fixed  = input_fixed.permute(0, 3, 1, 2)
    
    
    train_loss_scores = []
    val_loss_scores = []
    
    train_reconstruction_scores = []
    val_reconstruction_scores = []
    
    running_loss_losses = []
    running_recon_losses = []
    
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

        # Run the data through the model to produce warp and flow field
        warp, flow = model(input_moving, input_fixed)
        
        # Warp segment using flow
        moving_seg = torch.from_numpy(moving_seg).to(device).float()
        moving_seg = moving_seg.permute(0, 3, 1, 2)
        
        warp_seg = trf(moving_seg, flow).detach().cpu().numpy()

        vals, _ = dice(warp_seg, atlas_seg, labels=target_label_numbers, nargout=2)

        train_dice_acc.append(np.mean(vals))
        
        # Calculate loss
        recon_loss = sim_loss_fn(warp, input_fixed) 
        grad_loss = grad_loss_fn(flow)
        loss = recon_loss + reg_param * grad_loss
        
        running_recon_losses.append(recon_loss.item())
        running_loss_losses.append(loss.item())

        print("Train Epoch: %d | Loss: %f | Reconstruction Loss: %f"\
                 % (i, loss.item(), recon_loss.item()), flush=True)

        # Backwards and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 50 == 0:
              
            #val_loss_score, val_reconstruction_score = eval_net(gpu, model, val_file, batch_size, input_fixed, device, atlas_file, data_dir, data_loss, reg_param)
            
            d = test_net(gpu, model, atlas_file, val_file, data_dir, target_label_numbers)
            
            train_loss_scores.append(np.average(running_loss_losses))
            train_reconstruction_scores.append(np.average(running_recon_losses))
            
            #val_loss_scores.append(val_loss_score)
            #val_reconstruction_scores.append(val_reconstruction_score)
            
            val_dices.append(d)
            train_dices.append(np.average(train_dice_acc))
            
            running_recon_losses = []
            running_loss_losses = []
            
            train_dice_acc = []
            
    return train_loss_scores, train_reconstruction_scores, train_dices, val_dices
            

def eval_net(gpu, 
             model,
             val_file,
             batch_size, 
             input_fixed, 
             device, 
             atlas_file, 
             data_dir, 
             data_loss, 
             reg_param):
    
    model.eval()

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    # Test file and anatomical labels we want to evaluate
    val_file = open(val_file)
    val_strings = val_file.readlines()
    val_vol_names = [data_dir + x.strip() + ".npz" for x in val_strings]
    
    sim_loss_fn = losses.ncc_loss if data_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss
    
    val_example_gen = datagenerators.example_gen(val_vol_names, batch_size)
    
    running_recon = 0
    running_loss = 0

    for k in range(0, len(val_strings)):

         # Generate the moving images and convert them to tensors.
        moving_image = next(val_example_gen)[0]
        input_moving = torch.from_numpy(moving_image).to(device).float()

        input_moving = input_moving.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            warp, flow = model(input_moving, input_fixed)

        # Calculate loss
        recon_loss = sim_loss_fn(warp, input_fixed) 
        grad_loss = grad_loss_fn(flow)
        loss = recon_loss + reg_param * grad_loss
        
        running_loss += loss
        running_recon += recon_loss

        print("Val Epoch: %d | Loss: %f | Reconstruction Loss: %f"\
                 % (k, loss.item(), recon_loss.item()), flush=True)
    
    return running_loss/len(val_strings), running_recon/len(val_strings)


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

    # Test file and anatomical labels we want to evaluate
    val_file = open(val_file)
    val_strings = val_file.readlines()
    
    def create_vol_name(data_dir, x):
        vol_string = data_dir + x + ".npz"
        aseg_string = vol_string.replace("vols", "asegs").replace("norm","aseg")
        return vol_string + "," + aseg_string
        
    val_vol_names = [create_vol_name(data_dir, x.strip()) for x in val_strings]

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 32, 16, 16]

    # set up atlas tensor
    input_fixed  = torch.from_numpy(atlas_vol).to(device).float()
    input_fixed  = input_fixed.permute(0, 3, 1, 2)

    # Use this to warp segments
    
    trf = SpatialTransformer(atlas_vol.shape[1:-1], mode='nearest')
    trf.to(device)

    for k in range(0, len(val_vol_names)):

        vol_name, seg_name = val_vol_names[k].split(",")
        
        X_vol, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)

        input_moving  = torch.from_numpy(X_vol).to(device).float()
        input_moving  = input_moving.permute(0, 3, 1, 2)
        
        warp, flow = model(input_moving, input_fixed)

        # Warp segment using flow
        moving_seg = torch.from_numpy(X_seg).to(device).float()
        moving_seg = moving_seg.permute(0, 3, 1, 2)
        
        warp_seg = trf(moving_seg, flow).detach().cpu().numpy()

        vals, labels = dice(warp_seg, atlas_seg, labels=target_label_numbers, nargout=2)

        return np.mean(vals)
        
        