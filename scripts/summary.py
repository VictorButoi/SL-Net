from unet import UNet
from unet import TiedUNet
from torchsummary import summary
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = UNet(n_channels=1, n_classes=15, bilinear=True).to(device)
model2 = TiedUNet(in_channels=1, nshared=64, n_classes=15, bilinear=True).to(device)
summary(model1, input_size=(1, 160, 192))
summary(model2, input_size=(1, 160, 192))