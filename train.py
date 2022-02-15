from gc import callbacks
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils import *
from model import *
import argparse

def argparser():
    p = argparse.ArgumentParser()

    # arguments
    # train, valid, test data path
    p.add_argument('--train_path', type= str, default= './data(jpeg)/train')
    p.add_argument('--valid_path', type= str, default= './data(jpeg)/valid')
    p.add_argument('--test_path', type= str, default= './data(jpeg)/test')
    
    # data augmentation arguments
    p.add_argument('--rot_degree', type= float, default= 60,
                help= 'degree of RandomRotation')
    p.add_argument('--v_flip_prob', type= float, default= 0.5,
                help= 'probability of RandomVerticalFlip')
    p.add_argument('--h_flip_prob', type= float, default= 0.5,
                help= 'probability of RandomHorizontalFlip')
    p.add_argument('--distortion_scale', type= float, default= 0.3,
                help= 'distortion scale of RandomPerspective')
    p.add_argument('--distortion_prob', type= float, default= 0.3)
    
    # model configs
    p.add_argument('--in_channels', type= int, default= 1)
    p.add_argument('--out_channels', type= int, default= 1)
    p.add_argument('--lr', type= float, default= 1e-2)

    # trainig step configs
    p.add_argument('--max_epochs', type= int, default= 3)
    p.add_argument('--batch_size', type= int, default= 32)
    p.add_argument('--ds_num_workers', type= int, default= 4,
                    help= 'the number of workers in a Dataloader')
    p.add_argument('--patience', type= int, default = 3,
                help= 'patience of EarlyStopping')
    # model path
    p.add_argument('--model_name', type= str, default= 'prototype')
    p.add_argument('--checkpoints', type= str, default= './model/checkpoints')

    config = p.parse_args()
    return config

def main(config):
    # Data augmentation
    transform = transforms.Compose(
        [transforms.RandomRotation([-config.rot_degree, config.rot_degree]), 
        transforms.RandomVerticalFlip(config.h_flip_prob),
        transforms.RandomHorizontalFlip(config.v_flip_prob),
        transforms.RandomPerspective(distortion_scale= config.distortion_scale, p= config.distortion_prob),
        transforms.ToTensor()
        ])
    # Data setes
    train_datasets = BrainTumorDataSet(config.train_path, transform= transform)
    valid_datasets = BrainTumorDataSet(config.valid_path)
    train_ds = DataLoader(train_datasets, batch_size= config.batch_size, num_workers= config.ds_num_workers)
    valid_ds = DataLoader(valid_datasets, batch_size= config.batch_size, num_workers= config.ds_num_workers)

    # Model
    u_net = Unet(config.in_channels, config.out_channels, config.lr)

    # Train the model 
    gpus = torch.cuda.device_count()

    trainer = pl.Trainer(
            checkpoint_callback=True,
            logger=True,
            max_epochs=config.max_epochs, gpus=gpus,
            weights_save_path= './model',
            callbacks= [EarlyStopping(monitor="val_loss", patience= config.patience)]
    )

    # checkpoint_callback = ModelCheckpoint(
    #                                 monitor="val_loss",
    #                                 dirpath="./model",
    #                                 filename=config.model_name,
    #                                 mode="min"
    #                             )

    trainer.fit(u_net, 
                train_ds, valid_ds
                )

    
if __name__ == '__main__':
    config = argparser()
    main(config)

