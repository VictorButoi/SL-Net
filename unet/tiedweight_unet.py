import torch.nn.functional as F

from .unet_parts import *


class TiedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(TiedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        factor = 2 if bilinear else 1
        #auto
        self.batch_norm = nn.BatchNorm2d(64)
        self.ReLU = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(2)

        self.inc = DoubleConv(n_channels, 64)

        self.super_down_layer = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.super_down_layer_double = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.super_up_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        """INTRO CONV"""
        x1 = self.inc(x)

        """FIRST DOWN CONV"""
        x2 = self.maxPool(x1)
        x2 = self.super_down_layer(x2)
        x2 = self.batch_norm(x2)
        x2 = self.ReLU(x2)
        x2 = self.super_down_layer(x2)
        x2 = self.batch_norm(x2)
        x2 = self.ReLU(x2)

        """SECOND DOWN CONV"""
        x3 = self.maxPool(x2)
        x3 = self.super_down_layer(x3)
        x3 = self.batch_norm(x3)
        x3 = self.ReLU(x3)
        x3 = self.super_down_layer(x3)
        x3 = self.batch_norm(x3)
        x3 = self.ReLU(x3)

        """THREE DOWN CONV"""
        x4 = self.maxPool(x3)
        x4 = self.super_down_layer(x4)
        x4 = self.batch_norm(x4)
        x4 = self.ReLU(x4)
        x4 = self.super_down_layer(x4)
        x4 = self.batch_norm(x4)
        x4 = self.ReLU(x4)

        """FOUR DOWN CONV"""
        x5 = self.maxPool(x4)
        x5 = self.super_down_layer(x5)
        x5 = self.batch_norm(x5)
        x5 = self.ReLU(x5)
        x5 = self.super_down_layer(x5)
        x5 = self.batch_norm(x5)
        x5 = self.ReLU(x5)

        """FIRST UP CONV"""
        x5 = self.super_up_layer(x5)
        diffY = torch.tensor([x4.size()[2] - x5.size()[2]])
        diffX = torch.tensor([x4.size()[3] - x5.size()[3]])
        x5 = F.pad(x5, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x6 = torch.cat([x4, x5], dim=1)
        x6 = self.super_down_layer_double(x6)
        x6 = self.batch_norm(x6)
        x6 = self.ReLU(x6)
        x6 = self.super_down_layer(x6)
        x6 = self.batch_norm(x6)
        x6 = self.ReLU(x6)

        """TWO UP CONV"""
        x = self.super_up_layer(x6)
        diffY = torch.tensor([x3.size()[2] - x.size()[2]])
        diffX = torch.tensor([x3.size()[3] - x.size()[3]])
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x3, x], dim=1)
        x = self.super_down_layer_double(x)
        x = self.batch_norm(x)
        x = self.ReLU(x)
        x = self.super_down_layer(x)
        x = self.batch_norm(x)
        x = self.ReLU(x)

        """THREE UP CONV"""
        x = self.super_up_layer(x)
        diffY = torch.tensor([x2.size()[2] - x.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x.size()[3]])
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x], dim=1)
        x = self.super_down_layer_double(x)
        x = self.batch_norm(x)
        x = self.ReLU(x)
        x = self.super_down_layer(x)
        x = self.batch_norm(x)
        x = self.ReLU(x)

        """FOUR UP CONV"""
        x = self.super_up_layer(x)
        diffY = torch.tensor([x1.size()[2] - x.size()[2]])
        diffX = torch.tensor([x1.size()[3] - x.size()[3]])
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x], dim=1)
        x = self.super_down_layer_double(x)
        x = self.batch_norm(x)
        x = self.ReLU(x)
        x = self.super_down_layer(x)
        x = self.batch_norm(x)
        x = self.ReLU(x)

        logits = self.outc(x)
        return logits
