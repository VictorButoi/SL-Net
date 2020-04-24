from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

lookup_table ={0:0,2:1,3:2,5:3,6:4,10:5}

class BrainD(Dataset):
    def __init__(self, imgs_dir, masks_dir, label_numbers=None, inputT='npz', max_images=-1, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        if max_images == -1:
            self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        else:
            self.ids = []
            for i, file in enumerate(listdir(imgs_dir)):
                if i < max_images:
                    if not file.startswith('.'):
                        self.ids.append(splitext(file)[0])
                else:
                    break
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        self.load_npz = (inputT == 'npz')
        self.label_numbers = label_numbers

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        
        if self.load_npz:
            mask_file = glob(self.masks_dir + (idx[:-4] + "aseg") + '*')
            img_file = glob(self.imgs_dir + idx + '*')
        else:
            mask_file = glob(self.masks_dir + idx + '*')
            img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        if self.load_npz:
            mask = np.load(mask_file[0])['vol_data']
            img = np.load(img_file[0])['vol_data']
            assert img.size == mask.size, \
                f'Image and mask {idx} should be the same size, but are {img.shape} and {mask.shape}'

            mask = mask[ np.newaxis, ...]
            img = img[np.newaxis, ...]

            if not self.label_numbers == None:
                bad_labels = np.setdiff1d(np.unique(mask), self.label_numbers)
                for label in bad_labels:
                    mask[mask==label] = 0
                for label in self.label_numbers:
                    mask[mask==label] = lookup_table[label]
        else:
            mask = Image.open(mask_file[0])
            img = Image.open(img_file[0])
            assert img.size == mask.size, \
                f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

            img = self.preprocess(img, self.scale)
            mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
