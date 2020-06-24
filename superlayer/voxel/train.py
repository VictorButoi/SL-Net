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
          atlas_file,
          lr,
          n_iter,
          data_loss,
          model,
          reg_param, 
          batch_size,
          train_module=None):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    # Produce the loaded atlas with dims.:160x192x224.
    atlas_vol = np.load(atlas_file)['vol'][np.newaxis, ..., np.newaxis]

    # Get all the names of the training data
    
    train_file = open('/home/vib9/src/SL-Net/jupyter/partitions/train.txt')
    train_strings = train_file.readlines()
    train_vol_names = [data_dir + x.strip() + ".npz" for x in train_strings]
    
    model = mod
        
    model.to(device)

    # Set optimizer and losses
    opt = Adam(model.parameters(), lr=lr)

    sim_loss_fn = losses.ncc_loss if data_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # data generator
    train_example_gen = datagenerators.example_gen(train_vol_names, batch_size)

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
   
    for i in range(n_iter):
        model.train()

        # Generate the moving images and convert them to tensors.
        moving_image = next(train_example_gen)[0]
        input_moving = torch.from_numpy(moving_image).to(device).float()

        input_moving = input_moving.permute(0, 3, 1, 2)

        # Run the data through the model to produce warp and flow field
        warp, flow = model(input_moving, input_fixed)

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

        if i % 10000 == 0 and not i == 0:
            model.eval()
                
            val_loss_score, val_reconstruction_score = eval_net(gpu, model, batch_size, input_fixed, device, atlas_file, data_dir, data_loss, reg_param)
            
            train_loss_scores.append(np.average(running_loss_losses))
            train_reconstruction_scores.append(np.average(running_recon_losses))
            val_loss_scores.append(val_loss_score)
            val_reconstruction_scores.append(val_reconstruction_score)
            
            running_recon_losses = []
            running_loss_losses = []
            
    return train_loss_scores, train_reconstruction_scores, val_loss_scores, val_reconstruction_scores
            

def eval_net(gpu, 
             model, 
             batch_size, 
             input_fixed, 
             device, 
             atlas_file, 
             data_dir, 
             data_loss, 
             reg_param):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"

    # Test file and anatomical labels we want to evaluate
    val_file = open('/home/vib9/src/SL-Net/jupyter/partitions/val.txt')
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
