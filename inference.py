import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils import *
from model import *
from tqdm import tqdm
import os
from glob import glob
import sys
import csv

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type= str, required= True, 
                help= 'a path to model (.ckpt format)')
    p.add_argument('--model_type', type= int, default= 0,
                help= 'model type is either 0 for \'u-net\' or 1 for \'deep-u-net\' and 2 for \'resnet+u-net\'')
    p.add_argument('--test_path', type= str, default= './data(jpeg)/test',
                help= 'a path to test dataset')
    p.add_argument('--prediction_path', type= str, default= './data(jpeg)/mask_prediction',
                help= 'a path to save predictions')
    p.add_argument('--thr', type= float, default=0.5,
                help= 'threshold to generate a mask')
    p.add_argument('--batch_size', type= int, default= 32)
    p.add_argument('--in_channels', type= int, default= 1)
    p.add_argument('--out_channels', type= int, default= 1)
    p.add_argument('--is_validation', type= bool, default= True)
    
    config = p.parse_args()
    return config

def main(config):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # read moddel from a checkpoint
    check_point = torch.load(config.model_path)
    
    # Model
    if config.model_type == 0:
        model = Unet(config.in_channels, config.out_channels)
    elif config.model_type == 1:
        model = DeepUnet(config.in_channels, config.out_channels)
    elif config.model_type == 2:
        assert config.in_channels == 3, 'in_channels of resnet should be 3'
        model = ResUNet(config.in_channels, config.out_channels)
    else:
        print('the model is not ready yet...')
        sys.exit()
        
    model.load_state_dict(check_point['state_dict'])
    model.to(device)

    test_datasets = BrainTumorDataSet(config.test_path) if config.is_validation \
                         else BrainTumorDataSet(config.test_path, is_train= False)
    test_data_names = glob(os.path.join(config.test_path, 'images/*.jpeg'))
    temp = os.path.join(config.test_path,'images')
    pred_data_names = list(map(lambda x:x.replace(temp, config.prediction_path), test_data_names))
    test_ds = DataLoader(test_datasets, batch_size = config.batch_size, shuffle= False)
    
    # predictions
    print('making predictions ...')
    preds = []
    ious = []
    for batch_idx, (img, mask) in tqdm(enumerate(test_ds)):
        model.eval()
        with torch.no_grad():
            img = img.to(device)
            pred = F.sigmoid(model(img)).detach().cpu()
            pred[pred > config.thr] = 1.
            pred[pred < config.thr] = 0.
            preds.append(pred)
            if config.is_validation:
                ious.append(mask_intersection_over_union(mask, pred).item())
    
    preds = torch.cat(preds, dim=0) # N, 1, 512, 512
    print(f'converting torch to PIL images and save in {config.prediction_path}...')
    os.makedirs(config.prediction_path, exist_ok=True)
    tf = transforms.ToPILImage()
    for i, pred in tqdm(enumerate(preds)):
        img = tf(pred)
        img.save(pred_data_names[i], format= 'png')

    if config.is_validation:
        mean_ious = np.mean(ious)
        iou_path = os.path.join(config.prediction_path, 'mean_iou.txt')    
        with open(iou_path, 'w', newline='') as f: 
            writer = csv.writer(f)
            writer.writerow([mean_ious])    

if __name__ == '__main__':
    config = argparser()
    main(config)
