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
        self.super_up_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.super_up_layer.weight = self.super_down_layer.weight.t()

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        """INTRO CONV"""
        print(x.shape)
        x1 = self.inc(x)

        """FIRST DOWN CONV"""
        print(x1.shape)
        x2 = self.maxPool(x1)
        x2 = self.super_layer(x2)
        x2 = self.batch_norm(x2)
        x2 = self.ReLU(x2)
        x2 = self.super_layer(x2)
        x2 = self.batch_norm(x2)
        x2 = self.ReLU(x2)

        """SECOND DOWN CONV"""
        print(x2.shape)
        x3 = self.maxPool(x2)
        x3 = self.super_layer(x3)
        x3 = self.batch_norm(x3)
        x3 = self.ReLU(x3)
        x3 = self.super_layer(x3)
        x3 = self.batch_norm(x3)
        x3 = self.ReLU(x3)

        """THREE DOWN CONV"""
        print(x3.shape)
        x4 = self.maxPool(x3)
        x4 = self.super_layer(x4)
        x4 = self.batch_norm(x4)
        x4 = self.ReLU(x4)
        x4 = self.super_layer(x4)
        x4 = self.batch_norm(x4)
        x4 = self.ReLU(x4)

        """FOUR DOWN CONV"""
        print(x4.shape)
        x5 = self.maxPool(x4)
        x5 = self.super_layer(x5)
        x5 = self.batch_norm(x5)
        x5 = self.ReLU(x5)
        x5 = self.super_layer(x5)
        x5 = self.batch_norm(x5)
        x5 = self.ReLU(x5)

        print(x5.shape)

        """FIRST UP CONV"""
        x5 = self.super_up_layer(x5)
        diffY = torch.tensor([x4.size()[2] - x5.size()[2]])
        diffX = torch.tensor([x4.size()[3] - x5.size()[3]])
        x5 = F.pad(x5, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x6 = torch.cat([x4, x5], dim=1)
        x6 = self.super_layer(x6)
        print(x6.shape)

        """TWO UP CONV"""
        x = self.super_up_layer(x)
        diffY = torch.tensor([x3.size()[2] - x.size()[2]])
        diffX = torch.tensor([x3.size()[3] - x.size()[3]])
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x3, x], dim=1)
        x = self.super_layer(x)
        print(x.shape)

        """THREE UP CONV"""
        x = self.super_up_layer(x)
        diffY = torch.tensor([x2.size()[2] - x.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x.size()[3]])
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x], dim=1)
        x = self.super_layer(x)
        print(x.shape)

        """FOUR UP CONV"""
        x = self.super_up_layer(x)
        diffY = torch.tensor([x1.size()[2] - x.size()[2]])
        diffX = torch.tensor([x1.size()[3] - x.size()[3]])
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x], dim=1)
        x = self.super_layer(x)
        print(x.shape)

        logits = self.outc(x)
        return logits
