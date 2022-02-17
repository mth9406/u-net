from PIL import Image
from glob import glob
import os
from tqdm import tqdm
import cv2

# paths = [
#          './data/train/images',
#          './data/train/masks',
#          './data/valid/images',
#          './data/valid/masks',
#          './data/test/images',
#          './data/test/masks'         
# ]

# for path in paths:
#     imgs = glob(os.path.join(path, '*.png')) # path
#     imgs_names = list(map(lambda x:x.replace('.png', '.jpeg'), imgs)) #path
#     imgs_names = list(map(lambda x:x.replace('data', 'data(jpeg)'),imgs_names))
#     # print(imgs[:3])
#     # print(imgs_names[:3])
#     for i,img in tqdm(enumerate(imgs)):
#         src = Image.open(img)
#         src.save(imgs_names[i], 'jpeg')

# paths = [
#          './data(jpeg)/train/images',
#          './data(jpeg)/train/masks',
#          './data(jpeg)/valid/images',
#          './data(jpeg)/valid/masks',
#          './data(jpeg)/test/images',
#          './data(jpeg)/test/masks'         
# ]

# for path in paths:
#     imgs = glob(os.path.join(path, '*.jpeg'))
#     for img in tqdm(imgs):
#         src = cv2.imread(img, cv2.IMREAD_COLOR)
#         cv2.imwrite(img, src)

from PIL import Image
img = Image.open('./data(jpeg)/train/images/1.jpeg')

from torchvision import transforms
tf = transforms.ToTensor()
print(img.size)
tensor = tf(img)
print(tensor.shape)