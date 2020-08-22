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
    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        super(unet_core, self).__init__()

        self.full_size = full_size
        self.enc_nf = enc_nf
        self.dec_nf = dec_nf
        
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i-1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, dec_nf[0], int(dec_nf[1]/2)))  # 1

        if len(dec_nf) > 4:
            self.dec.append(conv_block(dim, dec_nf[1], int(dec_nf[2]/2)))  # 2
            self.dec.append(conv_block(dim, dec_nf[2], int(dec_nf[3]/2)))  # 3
            self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 4
            self.dec.append(conv_block(dim, dec_nf[4]+2, dec_nf[5]))  # 5
        else:
            self.dec.append(conv_block(dim, dec_nf[1], dec_nf[2]))  # 2
            self.dec.append(conv_block(dim, dec_nf[2]+2, dec_nf[3]))

        if self.full_size:
            self.dec.append(conv_block(dim, dec_nf[4] + 2, dec_nf[5], 1))
            self.vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6])
        
       
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x = l(x_enc[-1])
            x_enc.append(x)

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(len(self.dec)):
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i+2)]], dim=1)
            y = self.dec[i](y)
            
        # Two convs at full_size/2 res
       
        if self.full_size:
            y = self.dec[3](y)
            y = self.dec[4](y)

        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)

        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)

        return y


class cvpr2018_net(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for 
    the 2018 implementation of voxelmorph.
    """
    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(cvpr2018_net, self).__init__()

        dim = len(vol_size)
        
        self.unet_model = unet_core(dim, enc_nf, dec_nf, full_size)

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

class sln_unet_core(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """
    def __init__(self, 
                 dim, 
                 enc_nf, 
                 dec_nf, 
                 full_size=True, 
                 superblock_size=0, 
                 pt_head_weight=None,
                 pt_head_bias=None,
                 pt_tail_weight=None,
                 pt_tail_bias=None,
                 weight=None,
                 bias=None):
        
        super(sln_unet_core, self).__init__()

        self.full_size = full_size
        self.enc_nf = enc_nf
        self.dec_nf = dec_nf

        self.half_size = int(superblock_size/2)
 
        if weight is None:
            nd = Normal(0, 1e-5) 
            self.W = nn.Parameter(nd.sample((self.half_size, superblock_size,3,3)))
            self.b = nn.Parameter(torch.zeros(self.half_size))
        else:
            self.W = nn.Parameter(torch.from_numpy(weight))
            self.W.requires_grad = False
            self.b = nn.Parameter(torch.from_numpy(bias))
            self.b.requires_grad = False
            
        self.pt_hw = torch.from_numpy(pt_head_weight).cuda() if not pt_head_weight is None else None
        self.pt_hb = torch.from_numpy(pt_head_bias).cuda() if not pt_head_bias is None else None
        self.pt_tw = torch.from_numpy(pt_tail_weight).cuda() if not pt_tail_weight is None else None
        self.pt_tb = torch.from_numpy(pt_tail_bias).cuda() if not pt_tail_bias is None else None
        
        self.in_block = conv_block(dim, 2, self.half_size, train=True)
        
        self.down_block = conv_block(dim, train=False)
        self.up_block = conv_block(dim, train=False)
        
        self.out_conv = conv_block(dim, self.half_size + 2, self.half_size, train=True)
 
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.down = torch.nn.MaxPool2d(2,2)

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        
        for i in range(len(self.enc_nf)):
            xenc = self.down(x_enc[-1])
            if i == 0:
                x = self.in_block(xenc, self.pt_hw, self.pt_hb)
            else:
                x = self.down_block(xenc, self.W[:,:self.half_size,:,:], self.b)
            x_enc.append(x)

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(len(self.dec_nf)):
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i+2)]], dim=1)
            if i == range(len(self.dec_nf))[-1]:
                y = self.out_conv(y, self.pt_tw, self.pt_tb)
            else:
                y = self.up_block(y, self.W, self.b)
                
        return y

    
class sln_ae_core(nn.Module):
    
    def __init__(self, 
                 dim, 
                 enc_nf, 
                 dec_nf, 
                 full_size=True, 
                 superblock_size=0, 
                 pt_head_weight=None,
                 pt_head_bias=None,
                 pt_tail_weight=None,
                 pt_tail_bias=None,
                 weight=None,
                 bias=None):
        
        super(sln_ae_core, self).__init__()

        self.full_size = full_size
        self.enc_nf = enc_nf
        self.dec_nf = dec_nf
 
        if weight is None:
            nd = Normal(0, 1e-5) 
            self.W = nn.Parameter(nd.sample((superblock_size, superblock_size,3,3)))
            self.b = nn.Parameter(torch.zeros(superblock_size))
        else:
            self.W = nn.Parameter(torch.from_numpy(weight))
            self.W.requires_grad = False
            self.b = nn.Parameter(torch.from_numpy(bias))
            self.b.requires_grad = False
            
        self.pt_hw = torch.from_numpy(pt_head_weight).cuda() if not pt_head_weight is None else None
        self.pt_hb = torch.from_numpy(pt_head_bias).cuda() if not pt_head_bias is None else None
        self.pt_tw = torch.from_numpy(pt_tail_weight).cuda() if not pt_tail_weight is None else None
        self.pt_tb = torch.from_numpy(pt_tail_bias).cuda() if not pt_tail_bias is None else None
        
        self.in_block = conv_block(dim, 2, superblock_size, train=True)
        
        self.down_block = conv_block(dim, train=False)
        self.up_block = conv_block(dim, train=False)
        
        self.out_conv = conv_block(dim, superblock_size, superblock_size, train=True)
 
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.down = torch.nn.MaxPool2d(2,2)

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        
        for i in range(len(self.enc_nf)):
            xenc = self.down(x_enc[-1])
            if i == 0:
                x = self.in_block(xenc, self.pt_hw, self.pt_hb)
            else:
                x = self.down_block(xenc, self.W, self.b)
            x_enc.append(x)

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(len(self.dec_nf)):
            y = self.upsample(y)
            if i == range(len(self.dec_nf))[-1]:
                y = self.out_conv(y, self.pt_tw, self.pt_tb)
            else:
                y = self.up_block(y, self.W, self.b)
                
        return y


class sln_cvpr2018_net(nn.Module):
    
    def __init__(self,
                 vol_size, 
                 enc_nf, 
                 dec_nf, 
                 full_size=True,
                 superblock_size=0, 
                 pt_head_weight=None,
                 pt_head_bias=None,
                 pt_tail_weight=None,
                 pt_tail_bias=None,
                 pt_flow_weight=None,
                 pt_flow_bias=None,
                 pt_spatial_tfm=None,
                 weight=None,
                 bias=None,
                 mode="unet"):
       
        super(sln_cvpr2018_net, self).__init__()

        dim = len(vol_size)
        
        if mode=="unet":
            self.core_model = sln_unet_core(dim, 
                                            enc_nf, 
                                            dec_nf, 
                                            full_size, 
                                            superblock_size, 
                                            pt_head_weight=pt_head_weight,
                                            pt_head_bias=pt_head_bias,
                                            pt_tail_weight=pt_tail_weight,
                                            pt_tail_bias=pt_tail_bias,
                                            weight=weight,
                                            bias=bias)
        elif mode=="ae":
            self.core_model = sln_ae_core(dim, 
                                          enc_nf, 
                                          dec_nf, 
                                          full_size, 
                                          superblock_size, 
                                          pt_head_weight=pt_head_weight,
                                          pt_head_bias=pt_head_bias,
                                          pt_tail_weight=pt_tail_weight,
                                          pt_tail_bias=pt_tail_bias,
                                          weight=weight,
                                          bias=bias)
        else:
            raise ValueError("Not implemented")

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)      

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        if pt_flow_weight is None:
            self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        else:
            self.flow.weight = nn.Parameter(torch.from_numpy(pt_flow_weight))
            self.flow.weight.requires_grad = False
            
            
        if pt_flow_bias is None:
            self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        else:
            self.flow.bias = nn.Parameter(torch.from_numpy(pt_flow_bias))
            self.flow.bias.requires_grad = False
        
        self.spatial_transform = SpatialTransformer(size=[128,128],
                                                    pt_tfm=pt_spatial_tfm)


    def forward(self, src, tgt):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt], dim=1)
        x = self.core_model(x)
        flow = self.flow(x)
        y = self.spatial_transform(src, flow)

        return y, flow