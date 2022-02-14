import torchvision.transforms as transforms
from torch.utils.data import Dataset

import os
# import cv2
from PIL import Image

import sys
from glob import glob

class BrainTumorDataSet(Dataset):
    
    def __init__(self, data_dir = './data', transform= None):
        '''
        images_path: ./data/[train, valid, test]/images/*png
        masks_path: ./data/[train, valid, test]/masks/*png

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
             self._get_image(self.images_path[idx]), self._get_image(self.masks_path[idx])
        # transfrom numpy to tensor
        # and normalize it in (0~1) range.
        
        if self.transform is not None:
            image, mask = self.transform(image), self.transform(mask)

        else:
            convert_tensor = transforms.ToTensor()
            image, mask = convert_tensor(image), convert_tensor(mask)


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

