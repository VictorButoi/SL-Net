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

def add_repeat_layers(d, layers, dim1, num_rep):
    for _ in range(num_rep):
        layers.append(conv_block(d, dim1, dim1))
    return layers
    
    
class unet_core(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """
    def __init__(self, dim, enc_nf, dec_nf, follow_convs, full_size, conv_num):
        super(unet_core, self).__init__()

        self.full_size = full_size
        self.enc_nf = enc_nf
        self.dec_nf = dec_nf
        
        if conv_num is None:
            self.conv_repeats = [1]*(len(enc_nf) + len(dec_nf))
        else:
            self.conv_repeats = conv_num
        
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        
        for i in range(len(enc_nf)):
            prev_nf = 2 
            if i == 0:
                prev_nf = 2
            else:
                prev_nf = enc_nf[i-1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))
            self.enc = add_repeat_layers(dim, self.enc, enc_nf[i], self.conv_repeats[i] - 1)

        # Decoder functions
        self.dec = nn.ModuleList()
        
        self.dec.append(conv_block(dim, dec_nf[0] + enc_nf[-1], dec_nf[1]))  # 1
        self.dec = add_repeat_layers(dim, self.dec, dec_nf[1], self.conv_repeats[len(enc_nf)] - 1)

        self.dec.append(conv_block(dim, dec_nf[1] + enc_nf[-2], dec_nf[2]))
        self.dec = add_repeat_layers(dim, self.dec, dec_nf[2], self.conv_repeats[len(enc_nf) + 1] - 1)

        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[-3], dec_nf[3]))
        self.dec = add_repeat_layers(dim, self.dec, dec_nf[3], self.conv_repeats[len(enc_nf) + 2] - 1)

        self.dec.append(conv_block(dim, dec_nf[3] + 2, dec_nf[4]))
        self.dec = add_repeat_layers(dim, self.dec, dec_nf[4], self.conv_repeats[len(enc_nf) + 3] - 1)

        if self.full_size:
            self.cont = nn.ModuleList()
            
            self.cont.append(conv_block(dim, follow_convs[0], follow_convs[1]))
            self.cont.append(conv_block(dim, follow_convs[1], follow_convs[2]))
        
       
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        last_x = x
        conv_repeats = self.conv_repeats.copy()
        
        i = 0
        while(i < len(self.enc)):
            x = self.enc[i](last_x)
            conv_repeats[0] -= 1
            if conv_repeats[0] == 0:
                del conv_repeats[0]
                x_enc.append(x)
            last_x = x
            i += 1
            
        y = x_enc[-1]
        
        j = 0
        k = 0
        conv_rep_copy = conv_repeats.copy()
        while(j < len(self.dec)):
            if conv_rep_copy[0] == conv_repeats[0]:
                y = self.upsample(y)
                y = torch.cat([y, x_enc[-(k+2)]], dim=1)
                k += 1
            y = self.dec[j](y)
            conv_repeats[0] -= 1
            if conv_repeats[0] == 0:
                del conv_rep_copy[0]
                del conv_repeats[0]
            j += 1

        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.cont[0](y)
            y = self.cont[1](y)

        return y


class cvpr2018_net(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for 
    the 2018 implementation of voxelmorph.
    """
    def __init__(self, vol_size, enc_nf, dec_nf, follow_convs, conv_num=None, full_size=False):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(cvpr2018_net, self).__init__()

        dim = len(vol_size)
        
        self.unet_model = unet_core(dim, enc_nf, dec_nf, follow_convs, full_size, conv_num)

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
                 depth,
                 follow_convs, 
                 conv_num,
                 full_size, 
                 superblock_size, 
                 pt_head_weight,
                 pt_head_bias,
                 pt_tail_weight,
                 pt_tail_bias,
                 weight,
                 bias):
        
        super(sln_unet_core, self).__init__()

        self.full_size = full_size
        self.depth = depth

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
        
        if conv_num is None:
            self.conv_repeats = [1]*(2 + 2*depth)
        else:
            self.conv_repeats = conv_num
            
        if self.full_size:
            self.cont = nn.ModuleList()
            
            self.cont.append(conv_block(dim, follow_convs[0], follow_convs[1]))
            self.cont.append(conv_block(dim, follow_convs[1], follow_convs[2]))

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        count = 0
        
        for i in range(self.depth):
            xenc = self.down(x_enc[-1])
            for _ in range(self.conv_repeats[count]):
                if i == 0:
                    x = self.in_block(xenc, self.pt_hw, self.pt_hb)
                else:
                    x = self.down_block(xenc, self.W[:,:self.half_size,:,:], self.b)
            x_enc.append(x)
            count+=1

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(self.depth):
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i+2)]], dim=1)
            for _ in range(self.conv_repeats[count]):
                if i == (self.depth - 1):
                    y = self.out_conv(y, self.pt_tw, self.pt_tb)
                else:
                    y = self.up_block(y, self.W, self.b)
            count+=1
            
        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.cont[0](y)
            y = self.cont[1](y)
                
        return y

    
class sln_ae_core(nn.Module):
    
    def __init__(self, 
                 dim, 
                 depth,
                 follow_convs, 
                 full_size, 
                 conv_num, 
                 superblock_size, 
                 pt_head_weight,
                 pt_head_bias,
                 pt_tail_weight,
                 pt_tail_bias,
                 weight,
                 bias):
        
        super(sln_ae_core, self).__init__()

        self.full_size = full_size
        self.depth = depth
 
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
        
        if conv_num is None:
            self.conv_repeats = [1]*(2 + 2*depth)
        else:
            self.conv_repeats = conv_num
        
        if self.full_size:
            self.cont = nn.ModuleList()
            
            self.cont.append(conv_block(dim, follow_convs[0], follow_convs[1]))
            self.cont.append(conv_block(dim, follow_convs[1], follow_convs[2]))

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        count = 0
        
        for i in range(self.depth):
            xenc = self.down(x_enc[-1])
            for _ in range(self.conv_repeats[count]):
                if i == 0:
                    x = self.in_block(xenc, self.pt_hw, self.pt_hb)
                else:
                    x = self.down_block(xenc, self.W, self.b)
            x_enc.append(x)
            count+=1

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(self.depth):
            y = self.upsample(y)
            for _ in range(self.conv_repeats[count]):
                if i == (self.depth - 1):
                    y = self.out_conv(y, self.pt_tw, self.pt_tb)
                else:
                    y = self.up_block(y, self.W, self.b)
            count+=1
            
        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.cont[0](y)
            y = self.cont[1](y)
                
        return y


class sln_cvpr2018_net(nn.Module):
    
    def __init__(self,
                 vol_size, 
                 depth,
                 follow_convs=None,
                 full_size=False,
                 conv_num=None,
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
            self.core_model = sln_unet_core(dim=dim,
                                            depth=depth,
                                            follow_convs=follow_convs,
                                            full_size=full_size,
                                            conv_num=conv_num, 
                                            superblock_size=superblock_size,
                                            pt_head_weight=pt_head_weight,
                                            pt_head_bias=pt_head_bias,
                                            pt_tail_weight=pt_tail_weight,
                                            pt_tail_bias=pt_tail_bias,
                                            weight=weight,
                                            bias=bias)
        elif mode=="ae":
            self.core_model = sln_ae_core(dim=dim,
                                            depth=depth,
                                            follow_convs=follow_convs,
                                            full_size=full_size,
                                            conv_num=conv_num, 
                                            superblock_size=superblock_size,
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
        
        if mode=="ae":
            self.flow = conv_fn(superblock_size, dim, kernel_size=3, padding=1)
        else:
            self.flow = conv_fn(int(superblock_size/2), dim, kernel_size=3, padding=1)

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        if pt_flow_weight is None:
            self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        else:
            self.flow.weight = nn.Parameter(torch.from_numpy(pt_flow_weight))
            self.flow.weight.requires_grad = False
            
            
        if pt_flow_bias is None:
            self.flow.bias = nn.Parameter(nd.sample(self.flow.bias.shape))
        else:
            self.flow.bias = nn.Parameter(torch.from_numpy(pt_flow_bias))
            self.flow.bias.requires_grad = False
        
        self.spatial_transform = SpatialTransformer(vol_size)


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