import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch

import os
# import cv2
from PIL import Image
import numpy as np
import random

import sys
from glob import glob

class BrainTumorDataSet(Dataset):
    
    def __init__(self, data_dir = './data(jpeg)', transform= None):
        '''
        images_path: ./data/[train, valid, test]/images/*jpeg
        masks_path: ./data/[train, valid, test]/masks/*jpeg

        '''
        super(BrainTumorDataSet, self).__init__()

        # get images from image directories
        self.images_path = glob(os.path.join(data_dir, 'images/*.jpeg'))
        # self.images = self._get_images(self.images_path)
         
        self.masks_path = glob(os.path.join(data_dir, 'masks/*.jpeg'))
        # self.masks = self._get_images(self.masks_path, cv2.IMREAD_GRAYSCALE)
        
        self.transform = transform

    def __getitem__(self, idx):
        image, mask =\
             self._get_image(self.images_path[idx]), self._get_image(self.masks_path[idx])
        # transfrom numpy to tensor
        # and normalize it in (0~1) range.
        
        if self.transform is not None:
            # make seed
            seed = np.random.randint(2147483647)
            # apply this seed to both image and mask
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            
            random.seed(seed) 
            torch.manual_seed(seed)
            mask = self.transform(mask)

        else:
            convert_tensor = transforms.ToTensor()
            image = convert_tensor(image)
            mask = convert_tensor(mask)


        return image, mask

    def __len__(self):
        return len(self.images_path)

    def _get_image(self, path):
        image= Image.open(path)
        if image is None:
            print('--(!) Image load failed')
            sys.exit()
        h, w = image.size[:2]
        if h != 512 or w != 512:
            image = image.resize((512, 512))
        return image

# evaluation measure
def mask_intersection_over_union(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = intersection.sum()/ union.sum()
    return iou_score
    