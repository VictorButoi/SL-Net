import sys
sys.path.append("/home/vib9/src/SL-Net/superlayer/voxel")
sys.path.append("/home/vib9/src/voxelmorph-sandbox/pytorch/")
sys.path.append("/home/vib9/src/SL-Net/superlayer/")
sys.path.append("/home/vib9/src/voxelmorph/ext/medipy-lib/medipy")
sys.path.append("../voxel/")
sys.path.append("../../")
sys.path.append("../../scripts/")
sys.path.append("/home/vib9/src/voxelmorph/ext/medipy-lib/medipy")
sys.path.append("/home/vib9/src/voxelmorph/pytorch")
import os

from circle_utils import *
from metrics import dice

from superlayer.utils import dice_coeff
# python imports
import glob
import random
import warnings
from argparse import ArgumentParser

import datagenerators
import losses
from models import SpatialTransformer

import argparse
import logging

from numpy import random

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

import superlayer.utils

from scripts import train_net, get_args
from superlayer.models import cvpr2018_net, sln_cvpr2018_net

sys.path.append("/home/vib9/src/voxelmorph/pytorch")
import datagenerators
from models import SpatialTransformer
import math

from metrics import dice
from skimage.draw import circle

def show_distribution(data):
    data = list(map(lambda x : round(x,3), data))
    
    f = plt.figure(figsize=(15,5))
    ax = f.add_subplot(111)

    ax.set_title("Circle Registration")

    plt.hist(data, bins=100)   

    ax.set_xlabel("Mini-epochs")
    ax.set_ylabel("Loss Val")
    ax.legend()
    ax.grid()

def save_weights(model):
    np.save("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_headweight", model.core_model.in_block.main.weight.cpu().data)
    np.save("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_headbias", model.core_model.in_block.main.bias.cpu().data)

    np.save("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_tailweight", model.core_model.out_conv.main.weight.cpu().data)
    np.save("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_tailbias", model.core_model.out_conv.main.bias.cpu().data)
    
    np.save("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_128b", model.core_model.b.cpu().data)
    np.save("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_128W", model.core_model.W.cpu().data)

    np.save("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_flowweight", model.flow.weight.cpu().data)
    np.save("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_flowbias", model.flow.bias.cpu().data)

    np.save("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_spatialtransformgrid", model.spatial_transform.grid.cpu().data)
    
def plot_image_results(model):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = "cuda"
        
    model.to(device)
    
    model.eval()

    trf = SpatialTransformer([128,128], mode='nearest')
    trf.to(device)

    near_circle_gen = gen_near_pair()

    near_moving_image, near_fixed_image = next(near_circle_gen)

    input_moving = torch.from_numpy(near_moving_image).to(device).float()
    input_fixed = torch.from_numpy(near_fixed_image).to(device).float()

    warp, flow = model(input_moving, input_fixed)
    near_warp_seg = trf(input_moving, flow).detach().cpu()
    
    far_circle_gen = gen_far_pair()
    
    far_moving_image, far_fixed_image = next(far_circle_gen)
    
    input_moving = torch.from_numpy(far_moving_image).to(device).float()
    input_fixed = torch.from_numpy(far_fixed_image).to(device).float()

    warp, flow = model(input_moving, input_fixed)
    far_warp_seg = trf(input_moving, flow).detach().cpu()
    

    f, axarr = plt.subplots(2,3,figsize=(15,8))

    axarr[0,0].set_title("Near Pair Moving")
    axarr[0,0].imshow(near_moving_image.squeeze(0).squeeze(0))

    axarr[0,1].set_title("Near Moving Target")
    axarr[0,1].imshow(near_fixed_image.squeeze(0).squeeze(0))

    axarr[0,2].set_title("Near Moving Pred")
    axarr[0,2].imshow(near_warp_seg.squeeze(0).squeeze(0))

    axarr[1,0].set_title("Far Pair Moving")
    axarr[1,0].imshow(far_moving_image.squeeze(0).squeeze(0))

    axarr[1,1].set_title("Far Moving Target")
    axarr[1,1].imshow(far_fixed_image.squeeze(0).squeeze(0))

    axarr[1,2].set_title("Far Moving Pred")
    axarr[1,2].imshow(far_warp_seg.squeeze(0).squeeze(0))

def generate_train_pair(k=0.5):
    x = random.randint(20,105)
    y = random.randint(20,105)
    
    x2 = random.randint(20,105)
    y2 = random.randint(20,105)
    
    vol_shape=(128,128)
    cg = np.meshgrid(range(vol_shape[0]), range(vol_shape[1]), indexing='xy')
    
    img1, seg1 = draw_faded_circle(cg, center=(x, y), radius=10, slope=k)
    img2, seg2 = draw_faded_circle(cg, center=(x2, y2), radius=10, slope=k)
    
    img1 = img1 + gaussian_noise(img1)
    img2 = img2 + gaussian_noise(img2)
    
    return img1, img2, seg1, seg2

def generate_far_pair(k=0.5):
    x = random.randint(20,105)
    if x < 30 or x > 95:
        y = random.randint(20,105)
    else:
        y = random.randint(20,25)
        
    vol_shape=(128,128)
    cg = np.meshgrid(range(vol_shape[0]), range(vol_shape[1]), indexing='xy')
        
    img1, seg1 = draw_faded_circle(cg, center=(x, y), radius=10, slope=k)
    img2, seg2 = draw_faded_circle(cg, center=(abs(128 - x), abs(128 - y)), radius=10, slope=k)
    
    img1 = img1 + gaussian_noise(img1)
    img2 = img2 + gaussian_noise(img2)
    
    return img1, img2, seg1, seg2


def gaussian_noise(img):
    mean = 0
    sigma = 0.03
    noise = np.random.normal(mean, sigma, img.shape)
    
    return noise

def place_circle(img, x, y, size,k):
    # fill circle
    rr = circle(x, y, size, img.shape)
    for x_coord, y_coord in zip(rr[0],rr[1]):
        distance = np.linalg.norm(np.array([x,y]) - np.array([x_coord,y_coord]))
        modifier = 1/(1 + np.exp(-k*(-distance + 7)))
        img[x_coord][y_coord] = 255 * modifier

    return img

def add_circle(arr, x, y, size, fill=False):
    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
    circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    new_arr = np.logical_or(arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True)).astype(int)
    return new_arr

def gen_train_pair(batch_size=1):
    while True:
        arrays = np.asarray([generate_train_pair() for _ in range(batch_size)])

        img1, img2 = arrays[:,0,:,:].reshape((batch_size,1,128,128)), arrays[:,1,:,:].reshape((batch_size,1,128,128))
        seg1, seg2 = arrays[:,2,:,:].reshape((batch_size,1,128,128)), arrays[:,3,:,:].reshape((batch_size,1,128,128))

        yield img1, img2, seg1, seg2
        
def gen_far_pair(batch_size=1):
    while True:
        arrays = np.asarray([generate_far_pair() for _ in range(batch_size)])
        
        img1, img2 = arrays[:,0,:,:].reshape((batch_size,1,128,128)), arrays[:,1,:,:].reshape((batch_size,1,128,128))
        seg1, seg2 = arrays[:,2,:,:].reshape((batch_size,1,128,128)), arrays[:,3,:,:].reshape((batch_size,1,128,128))

        yield img1, img2, seg1, seg2
        
def logistic(x, x0=0., alpha=1.):
    return 1 / (1 + np.exp(-alpha * (x-x0)))
        
def draw_faded_circle(coord_grid, center, radius, slope=1):
    center = np.array(center).reshape((2,1,1))
    xv, yv = coord_grid
    image = np.stack((xv,yv))
    diff = image - center
    dst_from_center = np.linalg.norm(diff, ord=2, axis=0) 
    
    image = 1 - logistic(dst_from_center, x0=radius, alpha=slope)
    seg = np.where(dst_from_center <= radius, 1, 0)

    return image, seg

def validate_circle_train(mod, trf, circle_gen, batch_size, num_iters=0):
    
    mod.eval()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = "cuda"
    
    running_val_dices = []
    
    if num_iters == 0:
        num_iters = math.ceil(512/batch_size)

    for j in range(num_iters):
        moving_image, fixed_image, moving_seg, fixed_seg = next(circle_gen)

        input_moving = torch.from_numpy(moving_image).to(device).float()
        input_fixed = torch.from_numpy(fixed_image).to(device).float()

        warp, flow = mod(input_moving, input_fixed)
        
        moving_seg = torch.from_numpy(moving_seg).to(device).float()
        fixed_seg = torch.from_numpy(fixed_seg).to(device).float()
        
        warp_seg = trf(moving_seg, flow).detach().cpu()
        dice_score = dice_coeff(warp_seg,fixed_seg.detach().cpu())
        
        running_val_dices.append(dice_score)

        print("Val Epoch: %d | Dice Score: %f"\
         % (j, dice_score.item()), flush=True)
    
    return running_val_dices


def circle_train(mod,
          n_iter,
          batch_size,
          gpu='0',
          lr=1e-4,
          data_loss='mse',
          reg_param=0.01,
          train_module=None):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
        
    mod.to(device)

    # Set optimizer and losses
    opt = Adam(mod.parameters(), lr=lr)

    sim_loss_fn = losses.mse_loss
    grad_loss_fn = losses.gradient_loss
    
    train_dices = []
    wonky_dices = []
    
    running_train_dices = []
    
    trf = SpatialTransformer([128,128], mode='nearest')
    trf.to(device)
    
    train_circle_gen = gen_train_pair(batch_size=batch_size)
    far_circle_gen = gen_far_pair(batch_size=batch_size)
    
    for i in range(math.ceil(n_iter/batch_size)):
        mod.train()
        moving_image, fixed_image, moving_seg, fixed_seg = next(train_circle_gen)
        
        input_moving = torch.from_numpy(moving_image).to(device).float()
        input_fixed = torch.from_numpy(fixed_image).to(device).float()
        
        warp, flow = mod(input_moving, input_fixed)
        
        moving_seg = torch.from_numpy(moving_seg).to(device).float()
        fixed_seg = torch.from_numpy(fixed_seg).to(device).float()
        
        warp_seg = trf(moving_seg, flow).detach().cpu()
        
        dice_score = dice_coeff(warp_seg,fixed_seg.detach().cpu())
        running_train_dices.append(dice_score)    
        
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
   
        if i % int(1024/batch_size) == 0:
            running_val_dices_far = validate_circle_train(mod, trf, far_circle_gen, batch_size)
            
            train_dices.append(np.average(running_train_dices))
            wonky_dices.append(np.average(running_val_dices_far))
            
            running_train_dices = []
            
    return train_dices, wonky_dices

def compare_lines(data_list, names):
    domain = len(data_list[0])
    x_values = [i for i in range(domain)]

    f = plt.figure(figsize=(15,6))
    ax = f.add_subplot(111)

    ax.set_title("Circle Registration")
    ax.set_ylim([0,1.005])
    
    for i in range(len(data_list)):
        ax.plot(x_values, data_list[i], label=names[i])

    ax.set_xlabel("Mini-epochs")
    ax.set_ylabel("Loss Val")
    ax.legend()
    ax.grid()
    
def sln_load_model(model_dict, vol_size, nf_enc, nf_dec, super_size, retrain_flow=False, mode='unet'):
 
    head_weight = model_dict['core_model.in_block.main.weight'].cpu().numpy()
    head_bias = model_dict['core_model.in_block.main.bias'].cpu().numpy()

    tail_weight = model_dict['core_model.out_conv.main.weight'].cpu().numpy()
    tail_bias = model_dict['core_model.out_conv.main.bias'].cpu().numpy()

    super_block = model_dict['core_model.W'].cpu().numpy()
    
    print(super_block.shape)
    raise ValueError
    super_bias = model_dict['core_model.b'].cpu().numpy()
    
    if not retrain_flow:
        flow_weight = model_dict['flow.weight'].cpu().numpy()
        flow_bias = model_dict['flow.bias'].cpu().numpy()
    else:
        flow_weight = None
        flow_bias = None

    spatial_tfm = model_dict['spatial_transform.grid'].cpu().numpy()

    model = sln_cvpr2018_net(vol_size, 
                             nf_enc, 
                             nf_dec, 
                             superblock_size=super_size, 
                             pt_head_weight=head_weight,
                             pt_head_bias=head_bias,
                             pt_tail_weight=tail_weight,
                             pt_tail_bias=tail_bias,
                             pt_flow_weight=flow_weight,
                             pt_flow_bias=flow_bias,
                             pt_spatial_tfm=spatial_tfm,
                             weight=super_block,
                             bias=super_bias,
                             mode=mode)
    return model
    
"""
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = "cuda"

mod.to(device)
trf = SpatialTransformer([128,128], mode='nearest')
trf.to(device)
decent_circle_gen = gen_far_pair(batch_size=1)
train_wonkydice2 = validate_circle_train(mod, trf, decent_circle_gen, 1, num_iters=4096)
names = ["sln2d2u,bs:16,al:0.0005"]

train_wonkydice1 = [idm.numpy().item(0) for idm in train_wonkydice2]

np.save("/home/vib9/src/SL-Net/superlayer/models/superblocks/train_dice1", train_dice1)
np.save("/home/vib9/src/SL-Net/superlayer/models/superblocks/train_wonkydice1", train_wonkydice1)

save_weights(model)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = "cuda"

model.to(device)
trf = SpatialTransformer([128,128], mode='nearest')
trf.to(device)
far_circle_gen = gen_far_pair()
train_wonkydice2 = validate_circle_train(model, trf, far_circle_gen)

np.save("/home/vib9/src/SL-Net/superlayer/models/superblocks/train_wonkydice2", train_wonkydice2)

head_weight = np.load("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_headweight.npy")
head_bias = np.load("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_headbias.npy")

tail_weight = np.load("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_tailweight.npy")
tail_bias = np.load("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_tailbias.npy")

super_block = np.load("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_128W.npy")
super_bias = np.load("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_128b.npy")

flow_weight = np.load("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_flowweight.npy")
flow_bias = np.load("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_flowbias.npy")

spatial_tfm = np.load("/home/vib9/src/SL-Net/superlayer/models/superblocks/circle_spatialtransformgrid.npy")

nf_enc = [128, 128, 128]
nf_dec = [128, 128, 128]

model = sln_cvpr2018_net(vol_size, 
                         nf_enc, 
                         nf_dec, 
                         superblock_size=256, 
                         pt_head_weight=head_weight,
                         pt_head_bias=head_bias,
                         pt_tail_weight=tail_weight,
                         pt_tail_bias=tail_bias,
                         pt_flow_weight=flow_weight,
                         pt_flow_bias=flow_bias,
                         pt_spatial_tfm=spatial_tfm,
                         weight=super_block,
                         bias=super_bias)
"""