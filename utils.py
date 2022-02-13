import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import io
import sys
from glob import glob

class BrainTumorDataSet(Dataset):
    
    def __init__(self, data_dir = './data', transform= None):
        '''
        images_path: ./data/images/0a~.../images/*png
        masks_path: ./data/images/0a~.../masks/*png

        '''
        super(BrainTumorDataSet, self).__init__()

        # get images from image directories
        self.images_path = glob(os.path.join(data_dir, 'images/*.png'))
        # self.images = self._get_images(self.images_path)
         
        self.masks_path = glob(os.path.join(data_dir, 'masks/*.png'))
        # self.masks = self._get_images(self.masks_path, cv2.IMREAD_GRAYSCALE)
        
        self.transform = transform

    def __getitem__(self, idx):
        
        image, mask =\
             self._get_image(self.images_path[idx], cv2.IMREAD_COLOR), self._get_image(self.masks_path[idx], cv2.IMREAD_GRAYSCALE)
        # transfrom numpy to tensor
        # and normalize it in (0~1) range.
        convert_tensor = transforms.ToTensor()
        image, mask = convert_tensor(image)/255., convert_tensor(mask[:,:,None])/255.

        return image, mask

    def __len__(self):
        return len(self.images_path)

    def _get_image(self, path, code):
        image = cv2.imread(path, code)
        if image is None:
            print('--(!) Image load failed')
            sys.exit()
        h, w = image.shape[:2]
        if h != 512 or w != 512:
            image = cv2.resize(image, (512,512))
        return image

