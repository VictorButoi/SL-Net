from unet import UNet
from unet import TiedUNet
from torchsummary import summary
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
model = TiedUNet(n_channels=3, n_classes=1, bilinear=True).to(device)
summary(model, input_size=(3, 128, 191))