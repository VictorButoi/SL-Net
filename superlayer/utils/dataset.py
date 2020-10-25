from os.path import splitext
from os import listdir
import logging
from glob import glob

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

lookup_table = {2.0 : 1, 3.0 : 2, 16.0 : 3, 41.0 : 4, 42.0 : 5}

class BrainD(Dataset):
    def __init__(self, imgs_dir, masks_dir, id_file=None, label_numbers=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = []
        
        with open(id_file) as id_file: 
            for line in id_file:
                if not line.startswith('.'):
                    self.ids.append(line[:-1])            
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.label_numbers = label_numbers

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        
        mask_file = glob(self.masks_dir + (idx[:-4] + "aseg") + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, 'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, 'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        mask = np.load(mask_file[0])['vol_data']
        img = np.load(img_file[0])['vol_data']
        
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.shape} and {mask.shape}'

        mask = mask[np.newaxis, ...]
        img = img[np.newaxis, ...]
        
        bad_labels = np.setdiff1d(np.unique(mask), self.label_numbers)
        for label in bad_labels:
            mask[mask==label] = 0
        for label in self.label_numbers:
            mask[mask==label] = lookup_table[label]
        

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
