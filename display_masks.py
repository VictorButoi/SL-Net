import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from PIL import Image  
import PIL 

dfile = "/home/vib9/src/Pytorch-UNet/data/test_masks/ABIDE_50002_mri_talairach_aseg.npz"
image = np.load(dfile)["vol_data"]
img = Image.fromarray(image, 'L')
img.save('my.jpg')