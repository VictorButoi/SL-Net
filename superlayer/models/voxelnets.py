"""
*Preliminary* pytorch implementation.

Networks for voxelmorph model

In general, these are fairly specific architectures that were designed for the presented papers.
However, the VoxelMorph concepts are not tied to a very particular architecture, and we 
encourage you to explore architectures that fit your needs. 
see e.g. more powerful unet function in https://github.com/adalca/neuron/blob/master/neuron/models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal

from .unet_parts import SpatialTransformer
from .unet_parts import conv_block

class unet_core(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """
    def __init__(self, dim, enc_nf, dec_nf, full_size=True, superblock_size=0):
        super(unet_core, self).__init__()

        self.full_size = full_size
        self.enc_nf = enc_nf
        self.dec_nf = dec_nf
        
        self.vm2 = len(dec_nf) == 7
        
        if superblock_size == 0:
            self.train_W = False
            
            # Encoder functions
            self.enc = nn.ModuleList()
            for i in range(len(enc_nf)):
                prev_nf = 2 if i == 0 else enc_nf[i-1]
                self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))

            # Decoder functions
            self.dec = nn.ModuleList()
            self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
            self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
            self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
            self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
            self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5

            if self.full_size:
                self.dec.append(conv_block(dim, dec_nf[4] + 2, dec_nf[5], 1))

            self.vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6])
        
        else:
            self.train_W = True
            half_size = int(superblock_size/2)
            
            self.W = torch.nn.Parameter(torch.randn(half_size, superblock_size,3,3))
            self.W.requires_grad = True
            hW = self.W[:,:half_size,:,:]
            
            self.down = torch.nn.MaxPool2d(2,2)
            
            self.superblock_size = superblock_size
            
            self.in_block = conv_block(dim, 2, half_size)
            self.down_block = conv_block(dim, half_size, half_size, weight=hW)
            self.up_block = conv_block(dim, superblock_size, half_size, weight=self.W)
            self.out_conv = conv_block(dim, half_size + 2, half_size, weight=hW)
 
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        
        if not self.train_W:
            for l in self.enc:
                x_enc.append(l(x_enc[-1]))
        else:
            for i in range(len(self.enc_nf)):
                xenc = self.down(x_enc[-1])
                if i == 0:
                    x = self.in_block(xenc)
                else:
                    x = self.down_block(xenc)
                x_enc.append(x)

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            if self.train_W:
                y = self.upsample(y)
                y = torch.cat([y, x_enc[-(i+2)]], dim=1)
                y = self.up_block(y)
            else:
                y = self.dec[i](y)
                y = self.upsample(y)
                y = torch.cat([y, x_enc[-(i+2)]], dim=1)
            
        # Two convs at full_size/2 res
        if self.train_W:
            y = self.down_block(y)
            y = self.down_block(y)
        else:
            y = self.dec[3](y)
            y = self.dec[4](y)

        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.out_conv(y) if self.train_W else self.dec[5](y)

        # Extra conv for vm2
        if self.vm2:
            y = self.down_block(y) if self.train_W else self.vm2_conv(y)

        return y


class cvpr2018_net(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for 
    the 2018 implementation of voxelmorph.
    """
    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True, superblock_size=0):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(cvpr2018_net, self).__init__()

        dim = len(vol_size)
        
        self.unet_model = unet_core(dim, enc_nf, dec_nf, full_size, superblock_size)

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)      

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.spatial_transform = SpatialTransformer(vol_size)


    def forward(self, src, tgt):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        y = self.spatial_transform(src, flow)

        return y, flow