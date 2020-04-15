import torch
import numpy as np
import os
from PIL import Image

train_dir = '/home/gid-dalcaav/projects/neuron/data/t1_mix/proc/resize256-crop_x32/train/vols'

def get_onehot(asegs):
    subset_regs = [[0,0],   #Background
                   [13,52], #Pallidum   
                   [18,54], #Amygdala
                   [11,50], #Caudate
                   [3,42],  #Cerebral Cortex
                   [17,53], #Hippocampus
                   [10,49], #Thalamus
                   [12,51], #Putamen
                   [2,41],  #Cerebral WM
                   [8,47],  #Cerebellum Cortex
                   [4,43],  #Lateral Ventricle
                   [7,46],  #Cerebellum WM
                   [16,16]] #Brain-Stem
    dim1, dim2, dim3 = asegs.shape
    chs = 14
    one_hot = torch.zeros(1, chs, dim1, dim2, dim3)
    for i,s in enumerate(subset_regs):
        combined_vol = (asegs == s[0]) | (asegs == s[1]) 
        one_hot[:,i,:,:,:] = torch.from_numpy(combined_vol*1).float()
    mask = one_hot.sum(1).squeeze() 
    ones = torch.ones_like(mask)
    non_roi = ones-mask    
    one_hot[0,-1,:,:,:] = non_roi    
    assert one_hot.sum(1).sum() == dim1*dim2*dim3, 'One-hot encoding does not added up to 1'
    return one_hot

count = 0
num_skipped = 0
for vol in os.listdir(train_dir):
    if not "MR2" in vol:
        file_name = train_dir + "/" + vol
        print(file_name)
        aseg = np.load(file_name)
        aseg = aseg['vol_data'].astype('float32')
        onehot = get_onehot(aseg).numpy()

        img = aseg[:,:,200]
        mask = onehot[0,0,:,:,200]

        im = Image.fromarray(img)
        mask = Image.fromarray(mask)

        im = im.convert("L")
        mask = mask.convert("L")
        
        im.save("data/imgs/" + vol[:-4] + ".jpg")
        mask.save("data/masks/" + vol[:-4] + ".png")

        count+=1
        
        print("Done :" + str(count))
        print("Percent Done :" + str(100 * (count/len(os.listdir(train_dir)))) + "%")
    else:
        num_skipped+=1

print("Skipped :" + str(num_skipped))