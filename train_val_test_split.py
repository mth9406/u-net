import numpy as np
from glob import glob
import os
import shutil

# (1) randomly shuffle the data
image_path = glob('./data/images/*.png')
mask_path = glob('./data/masks/*.png')
num_obs = len(image_path)

# randomly shuffled indices
idx = np.random.permutation(np.arange(num_obs))
# split the indices in tr:val:te = 7:2:1
endIdxTr = int(num_obs * 0.7)
endIdxVal = endIdxTr + int(num_obs * 0.2)

# (2) split data into train:val:test
image_path_tr, image_path_val, image_path_te =\
    image_path[:endIdxTr], image_path[endIdxTr:endIdxVal], image_path[endIdxVal:]
mask_path_tr, mask_path_val, mask_path_te =\
    mask_path[:endIdxTr], mask_path[endIdxTr:endIdxVal], mask_path[endIdxVal:]

# (3) make directories to save the images

folder_list = [
    './data/train/images',
    './data/train/masks',
    './data/valid/images',
    './data/valid/masks',
    './data/test/images',
    './data/test/masks'
]

sources = [
    image_path_tr, 
    mask_path_tr,
    image_path_val, 
    mask_path_val,
    image_path_te,
    mask_path_te
]


for folder in folder_list:
    os.mkdir(folder)

for source, folder in zip(sources, folder_list):
    for s in source:
        shutil.move(s, folder)


    
