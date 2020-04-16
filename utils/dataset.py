from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BrainD(Dataset):
    def __init__(self, imgs_dir, masks_dir, inputT='npz', max_images=-1, scale=1):
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
        mask_file = glob(self.masks_dir + (idx[:-4] + "aseg") + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        print(mask_file)
        print(img_file)

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        if self.load_npz:
            mask = np.load(mask_file[0]).numpy()
            img = np.load(img_file[0]).numpy()
            assert img.size == mask.size, \
                f'Image and mask {idx} should be the same size, but are {img.shape} and {mask.shape}'
        else:
            mask = Image.open(mask_file[0])
            img = Image.open(img_file[0])
            assert img.size == mask.size, \
                f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

            img = self.preprocess(img, self.scale)
            mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
