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
        print(x.shape)
        x1 = self.inc(x)

        down_path = [x1]

        for i in range(4):
            down_path.append(self.conv_seq(down_path[-1]))

        up_path = [down_path[-1]]
        for i in range(2,6):
            x1 = up_path[-1]
            x2 = down_path[-i]
            x1 = self.super_up_layer(x1)
            diffY = torch.tensor(x2.size()[2] - x1.size()[2])
            diffX = torch.tensor(x2.size()[3] - x1.size()[3])
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            up_path.append(self.conv_seq(x, option='up'))

        logits = self.outc(up_path[-1])
        return logits
