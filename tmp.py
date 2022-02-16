import torch

from PIL import Image
from glob import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


paths = [
         './data/train/images',
         './data/train/masks',
         './data/valid/images',
         './data/valid/masks',
         './data/test/images',
         './data/test/masks'         
]

for path in paths:
    imgs = glob(os.path.join(path, '*.png')) # path
    imgs_names = list(map(lambda x:x.replace('.png', '.jpeg'), imgs)) #path
    imgs_names = list(map(lambda x:x.replace('data', 'data(jpeg)'),imgs_names))
    # print(imgs[:3])
    # print(imgs_names[:3])
    for i,img in tqdm(enumerate(imgs)):
        src = Image.open(img)
        src.save(imgs_names[i], 'jpeg')