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
    p.add_argument('--model_type', type= int, default= 0,
                help= 'model type is either 0 for \'u-net\' or 1 for \'deep-u-net\' and 2 for \'resnet+u-net\'')
    p.add_argument('--is_continue_training', type= bool, default= False,
                help= 'set \'True\' if you are to continue training')
    p.add_argument('--model_path', type= str, required= False,
                help= 'saved model path to resume training (.ckpt format)')
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
    p.add_argument('--checkpoints', type= str, default= './model')

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
    if config.model_type == 0:
        u_net = Unet(config.in_channels, config.out_channels, config.lr)
    elif config.model_type == 1:
        u_net = DeepUnet(config.in_channels, config.out_channels, config.lr)
    elif config.model_type == 2:
        assert config.in_channels == 3, 'in_channels of resnet should be 3'
        u_net = ResUNet(config.in_channels, config.out_channels, config.lr)
    else:
        print('the model is not ready yet...')
        sys.exit()
    
    if config.is_continue_training:
        check_point = torch.load(config.model_path)
        u_net.load_state_dict(check_point['state_dict'])

    # Train the model 
    gpus = torch.cuda.device_count()

    checkpoint_callback = ModelCheckpoint(
                                    dirpath= config.checkpoints,
                                    filename= '{epoch}-{val_loss:.4f}',
                                    monitor="val_loss",
                                    mode="min"
                                )

    trainer = pl.Trainer(
            checkpoint_callback=True,
            logger=True,
            max_epochs=config.max_epochs, gpus=gpus,
            # weights_save_path= config.checkpoints,
            callbacks= [checkpoint_callback, EarlyStopping(monitor="val_loss", patience= config.patience)]
    )

    trainer.fit(u_net, 
                train_ds, valid_ds
                )

    
if __name__ == '__main__':
    config = argparser()
    main(config)

