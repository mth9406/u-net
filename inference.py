import argparse
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from utils import *
from model import *
from tqdm import tqdm
import os
from glob import glob

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type= str, required= True, 
                help= 'a path to model (.ckpt format)')
    p.add_argument('--test_path', type= str, default= './data/test',
                help= 'a path to test dataset')
    p.add_argument('--prediction_path', type= str, default= './data/mask_prediction',
                help= 'a path to save predictions')
    p.add_argument('--batch_size', type= int, default= 32)
    p.add_argument('--in_channels', type= int, default= 1)
    p.add_argument('--out_channels', type= int, default= 1)
    
    config = p.parse_args()
    return config

def main(config):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # read moddel from a checkpoint
    check_point = torch.load(config.model_path)
    
    model = Unet(config.in_channels, config.out_channels)
    model.load_state_dict(check_point['state_dict'])
    model.to(device)

    test_datasets = BrainTumorDataSet(config.test_path)
    test_data_names = glob(os.path.join(config.test_path, 'images/*.png'))
    temp = os.path.join(config.test_path,'images')
    pred_data_names = list(map(lambda x:x.replace(temp, config.prediction_path), test_data_names))
    test_ds = DataLoader(test_datasets, batch_size = config.batch_size, shuffle= False)
    
    # predictions
    print('making predictions ...')
    preds = []
    # socres = []
    for batch_idx, (img, mask) in tqdm(enumerate(test_ds)):
        model.eval()
        with torch.no_grad():
            img = img.to(device)
            pred = model(img)
            preds.append(pred.detach())
    
    preds = torch.cat(preds, dim=0) # N, 1, 512, 512
    print(f'converting torch to PIL images and save in {config.prediction_path}...')
    os.makedirs(config.prediction_path, exist_ok=True)
    tf = transforms.ToPILImage()
    for i, pred in tqdm(enumerate(preds)):
        img = tf(pred)
        img.save(pred_data_names[i], format= 'png')
    
if __name__ == '__main__':
    config = argparser()
    main(config)