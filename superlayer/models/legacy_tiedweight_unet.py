import torch.nn.functional as F
import sys

from .unet_parts import *

class TiedUNet(nn.Module):
    def __init__(self, in_channels, nshared, n_classes, enc_depth, bilinear=True):
        super(TiedUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_downsizes = enc_depth
        
        self.batch_norm = nn.BatchNorm2d(nshared)
        self.ReLU = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(2)

        self.inc = DoubleConv(in_channels, nshared)

        self.super_down_layer = nn.Conv2d(nshared, nshared, kernel_size=3, padding=1)
        self.super_down_layer_double = nn.Conv2d(2*nshared, nshared, kernel_size=3, padding=1)
        self.super_up_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(nshared, n_classes, kernel_size=1)
        self.sm = nn.Softmax(dim=1)

    
    def conv_seq(self, x, option='down'):
        if(option == 'down'):
            x = self.maxPool(x)
            x = self.super_down_layer(x)
        else:
            x = self.super_down_layer_double(x)
        x = self.batch_norm(x)
        x = self.ReLU(x)
        x = self.super_down_layer(x)
        x = self.batch_norm(x)
        x = self.ReLU(x)
        return x


    def forward(self, x):
        x1 = self.inc(x)

        down_path = [x1]

        for i in range(self.n_downsizes):
            down_path.append(self.conv_seq(down_path[-1]))

        up_path = [down_path[-1]]
        for i in range(self.n_downsizes):
            x1 = up_path[-1]
            x2 = down_path[-(i+2)]
            x1 = self.super_up_layer(x1)
            diffY = torch.tensor(x2.size()[2] - x1.size()[2])
            diffX = torch.tensor(x2.size()[3] - x1.size()[3])
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            up_path.append(self.conv_seq(x, option='up'))

        logits = self.outc(up_path[-1])
        out = self.sm(logits)
        return out