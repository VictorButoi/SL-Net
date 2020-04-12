from unet import UNet
from unet import TiedUNet
from torchsummary import summary

model = UNet(n_channels=3, n_classes=1, bilinear=True)
#model = TiedUNet(n_channels=3, n_classes=1, bilinear=True)
summary(model, input_size=(3, 128, 191))