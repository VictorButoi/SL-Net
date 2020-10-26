""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    
class simple_block(nn.Module):
    def __init__(self, in_channels=0, out_channels=0, use_bn=True, weight=None, bias=None, sb_size=0, train_block=False):
        super(simple_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sb_size = sb_size
        self.sb_halfsize = int(sb_size/2)

        self.use_bn= use_bn
        
        if weight is None and train_block:
            nd = Normal(0, 1e-5) 
            self.W = nn.Parameter(nd.sample((self.sb_halfsize, sb_size, 3, 3)))
            self.b = nn.Parameter(torch.zeros(self.sb_halfsize))  
            self.use_weight = True
        elif not(weight is None):
            self.W = weight
            self.b = bias
            self.use_weight = True
        else:
            self.use_weight = False
            
        
        if self.use_weight == False:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.InstanceNorm2d(out_channels)  
        self.activation = nn.ReLU()

    def forward(self, x, use_half=False):
        if self.use_weight:
            if use_half:
                out = F.conv2d(x, self.W[:,:self.sb_halfsize,:,:], bias=self.b, padding=1)
            else:
                out = F.conv2d(x, self.W, bias=self.b, padding=1)
        else:  
            out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.activation(out)
        return out
    
class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, size, pt_tfm=None, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [ torch.arange(0, s) for s in size ] 
        grids = torch.meshgrid(vectors) 
        if pt_tfm is None:
            grid = torch.stack(grids) # y, x, z
            grid = torch.unsqueeze(grid, 0)  #add batch
            grid = grid.type(torch.FloatTensor)
        else:
            grid = torch.from_numpy(pt_tfm)
        
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):   
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow 

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1) 
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1) 
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=self.mode)
    

class FeatureWeighter(nn.Module):
    def __init__(self, weight, bias):
        """
        Instiatiate the block
            :param weight: the premade weight block
        """
        super(FeatureWeighter, self).__init__()
        self.weight = weight
        self.bias = bias
        
        w_shape = weight.shape
        nd = Normal(0, 1e-5) 
        self.multiplier = nn.Parameter(nd.sample((w_shape[0],w_shape[1],1,1)))

    def forward(self, src):   
        new_weight = self.multiplier * self.weight
        return F.conv2d(src, new_weight, bias=self.bias, padding=1)


class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """
    def __init__(self, dim, in_channels=0, out_channels=0, stride=1, train=True):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()
        
        Conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')
        
        if train:
            self.main = Conv_fn(in_channels, out_channels, ksize, stride, 1)
        
        self.dim = dim
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, W=None, b=None):
        """
        Pass the input through the conv_block
        """
        if W is None:
            out = self.main(x)
        else:
            conv_fn = getattr(F, "conv{0}d".format(self.dim))
            out = conv_fn(x, W, b, stride=1, padding=1)

        out = self.activation(out)
        return out