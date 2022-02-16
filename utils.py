import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch

import os
from PIL import Image
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import random

import sys
from glob import glob


def unet_weight_map(y, w0 = 10, sigma = 5):

    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.
    """

    def find_wc(m):
        totPixels = m.shape[0]*m.shape[1]
        roiPixels = torch.sum(m==1.)
        backgPixels = totPixels - roiPixels

        wc = {
            0.:w0*roiPixels/totPixels,
            1.:w0*backgPixels/totPixels
        }
        return wc

    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
    else:
        w = np.zeros_like(y)

    wc = find_wc(y)

    class_weights = np.zeros_like(y)
    for k, v in wc.items():
        class_weights[y == k] = v
    w = torch.FloatTensor(w + class_weights)
    
    return w

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
        if transform is None:
            self.convert_tensor = transforms.ToTensor()


    def __getitem__(self, idx):
        image, mask =\
             self._get_image(self.images_path[idx]), self._get_image(self.masks_path[idx], convert= True)
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
            image = self.convert_tensor(image)
            mask = self.convert_tensor(mask)

        return image, mask


    def __len__(self):
        return len(self.images_path)

    def _get_image(self, path, convert= False):
        image= Image.open(path)
        if image is None:
            print('--(!) Image load failed')
            sys.exit()
        h, w = image.size[:2]
        if h != 512 or w != 512:
            image = image.resize((512, 512))
        if convert:
            image = image.convert('L')
        return image

# evaluation measure
def mask_intersection_over_union(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = intersection.sum()/ union.sum()
    return iou_score
    