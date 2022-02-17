import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import *
from torchvision import models

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (3,3), padding= 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, (3,3), padding= 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)

class DoubleConvResidBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConvResidBlock, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (3,3), padding= 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace= True),
                nn.Conv2d(out_channels, out_channels, (3,3), padding= 1),
                nn.BatchNorm2d(out_channels)
        )
        self.mapping = nn.Conv2d(in_channels, out_channels, (1,1))
        
    def forward(self, x):
        out = self.block(x)
        out = out + self.mapping(x)
        out = F.relu(out, inplace= True)
        return out

# Original U-net with residual blocks
class Unet(pl.LightningModule):

    def __init__(self, in_channels, out_channels, 
                lr=1e-3):
        super(Unet, self).__init__()
        
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = DoubleConv(in_channels, 32)
        self.conv2 = DoubleConv(32, 64)
        self.conv3 = DoubleConv(64, 128)
        self.conv4 = DoubleConv(128, 256)
        self.conv5 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # halves the H X W
        
        self.up1 = nn.ConvTranspose2d(in_channels = 512, 
                    out_channels =256, kernel_size = 2, stride= 2)
        self.up2 = nn.ConvTranspose2d(in_channels = 256, 
                    out_channels =128, kernel_size = 2, stride= 2)
        self.up3 = nn.ConvTranspose2d(in_channels = 128, 
                    out_channels =64, kernel_size = 2, stride= 2)
        self.up4 = nn.ConvTranspose2d(in_channels = 64, 
                    out_channels =32, kernel_size = 2, stride= 2)
        
        self.up_conv1 = DoubleConv(512, 256)
        self.up_conv2 = DoubleConv(256, 128)
        self.up_conv3 = DoubleConv(128, 64)
        self.up_conv4 = DoubleConv(64, 32)

        self.decode = nn.Sequential(
            nn.Conv2d(32, out_channels, 1),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        # encoder
        # down sampling
        c1 = self.conv1(x) # 3, 512, 512 -> 64, 512, 512
        p1 = self.pool(c1) # -> 64, 256, 256

        c2 = self.conv2(p1) # 64, 256, 256 
        p2 = self.pool(c2) # -> -> 128, 128, 128

        c3 = self.conv3(p2) # 128, 128, 128
        p3 = self.pool(c3) # -> -> 256, 64, 64

        c4 = self.conv4(p3) # 256, 64, 64
        p4 = self.pool(c4) # -> -> 512, 32, 32

        c5 = self.conv5(p4) # 512, 32, 32

        u1 = self.up1(c5) # 512, 32, 32
        cat1 = torch.cat([u1, c4], dim = 1) # 1024, 32, 32
        uc1 = self.up_conv1(cat1) # 512, 32, 32

        u2 = self.up2(uc1) # 256, 64, 64
        cat2 = torch.cat([u2, c3], dim = 1) # 512, 64, 64
        uc2 = self.up_conv2(cat2) # 256, 64, 64

        u3 = self.up3(uc2) # 128, 128, 128
        cat3 =  torch.cat([u3, c2], dim = 1) # 256, 128, 128
        uc3 = self.up_conv3(cat3) # 128, 256, 256 

        u4 = self.up4(uc3) # 64, 512, 512 
        cat4 = torch.cat([u4, c1], dim= 1) # 128, 512, 612
        uc4 = self.up_conv4(cat4) # 64, 512, 512

        outputs = self.decode(uc4) # out_channels(=1), 512, 512 
        
        return outputs

    def predict(self, x, thr= 0.5):
        p = self(x)
        p[p > thr] = 255 
        p[p < thr] = 0
        return p

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step= True, 
                    on_epoch= True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_step= True, 
                    on_epoch= True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr= self.lr)

# Deep Unet    
class DeepUnet(pl.LightningModule):

    def __init__(self, in_channels, out_channels, 
                lr=1e-3):
        super(DeepUnet, self).__init__()
        
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = DoubleConvResidBlock(in_channels, 4)
        self.conv2 = DoubleConvResidBlock(4, 8)
        self.conv3 = DoubleConvResidBlock(8, 16)
        self.conv4 = DoubleConvResidBlock(16, 32)
        self.conv5 = DoubleConvResidBlock(32, 64)
        self.conv6 = DoubleConvResidBlock(64, 128)
        self.conv7 = DoubleConvResidBlock(128, 256)


        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # halves the H X W
        
        self.up1 = nn.ConvTranspose2d(in_channels = 256, 
                    out_channels =128, kernel_size = 2, stride= 2)
        self.up2 = nn.ConvTranspose2d(in_channels = 128, 
                    out_channels =64, kernel_size = 2, stride= 2)
        self.up3 = nn.ConvTranspose2d(in_channels = 64, 
                    out_channels =32, kernel_size = 2, stride= 2)
        self.up4 = nn.ConvTranspose2d(in_channels = 32, 
                    out_channels =16, kernel_size = 2, stride= 2)
        self.up5 = nn.ConvTranspose2d(in_channels = 16, 
                    out_channels =8, kernel_size = 2, stride= 2)
        self.up6 = nn.ConvTranspose2d(in_channels = 8, 
                    out_channels =4, kernel_size = 2, stride= 2)

        self.up_conv1 = DoubleConvResidBlock(256, 128)
        self.up_conv2 = DoubleConvResidBlock(128, 64)
        self.up_conv3 = DoubleConvResidBlock(64, 32)
        self.up_conv4 = DoubleConvResidBlock(32, 16)
        self.up_conv5 = DoubleConvResidBlock(16, 8)
        self.up_conv6 = DoubleConvResidBlock(8, 4)
        
        self.decode = nn.Sequential(
            nn.Conv2d(4, out_channels, 1),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        # encoder
        # down sampling
        c1 = self.conv1(x) # 3, 512, 512 -> 64, 512, 512
        p1 = self.pool(c1) # -> 64, 256, 256

        c2 = self.conv2(p1) # 64, 256, 256 
        p2 = self.pool(c2) # -> -> 128, 128, 128

        c3 = self.conv3(p2) # 128, 128, 128
        p3 = self.pool(c3) # -> -> 256, 64, 64

        c4 = self.conv4(p3) # 256, 64, 64
        p4 = self.pool(c4) # -> -> 512, 32, 32

        c5 = self.conv5(p4) # 512, 32, 32
        p5 = self.pool(c5)

        c6 = self.conv6(p5)
        p6 = self.pool(c6)

        c7 = self.conv7(p6)

        u1 = self.up1(c7) # 512, 32, 32
        cat1 = torch.cat([u1, c6], dim = 1) # 1024, 32, 32
        uc1 = self.up_conv1(cat1) # 512, 32, 32

        u2 = self.up2(uc1) # 256, 64, 64
        cat2 = torch.cat([u2, c5], dim = 1) # 512, 64, 64
        uc2 = self.up_conv2(cat2) # 256, 64, 64

        u3 = self.up3(uc2) # 128, 128, 128
        cat3 =  torch.cat([u3, c4], dim = 1) # 256, 128, 128
        uc3 = self.up_conv3(cat3) # 128, 256, 256 

        u4 = self.up4(uc3) # 64, 512, 512 
        cat4 = torch.cat([u4, c3], dim= 1) # 128, 512, 612
        uc4 = self.up_conv4(cat4) # 64, 512, 512

        u5 = self.up5(uc4) # 64, 512, 512 
        cat5 = torch.cat([u5, c2], dim= 1) # 128, 512, 612
        uc5 = self.up_conv5(cat5) # 64, 512, 512

        u6 = self.up6(uc5) # 64, 512, 512 
        cat6 = torch.cat([u6, c1], dim= 1) # 128, 512, 612
        uc6 = self.up_conv6(cat6) # 64, 512, 512


        outputs = self.decode(uc6) # out_channels(=1), 512, 512 
        
        return outputs

    def predict(self, x, thr= 0.5):
        p = self(x)
        p[p > thr] = 255 
        p[p < thr] = 0
        return p

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step= True, 
                    on_epoch= True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        return y_hat 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr= self.lr)

# U-net using Resnet as a backbone network   
class ResUNet(pl.LightningModule):

    def __init__(self, in_channels, out_channels, 
                lr=1e-3):
        super(ResUNet, self).__init__()
        
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.backbone = models.resnet34(pretrained= True)
        self.base_layers = list(self.backbone.children())

        # encoder layer (resnet)
        # assume input image batch's shape = bs, 3, 512, 512
        self.enc0 = nn.Sequential(*self.base_layers[:3]) # bs, 64, 256, 256
        self.enc1 = nn.Sequential(*self.base_layers[3:5]) # bs, 64, 128, 128
        self.enc2 = nn.Sequential(*self.base_layers[5]) # bs, 128, 64, 64
        self.enc3 = nn.Sequential(*self.base_layers[6]) # bs, 256, 32, 32
        self.enc4 = nn.Sequential(*self.base_layers[7]) # bs, 512, 16, 16

        # decoder layer
        # up-sample
        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2) # bs, 256, 32, 32
        # concat up1 and enc3: bs, 256+256, 32, 32
        self.conv1 = DoubleConvResidBlock(256+256, 256) # bs, 256, 32, 32 
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2) # bs, 128, 64, 64
        # concat up2 and enc2: bs, 128+128, 64, 64
        self.conv2 = DoubleConvResidBlock(128+128, 128) # bs, 128, 64, 64
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2) # bs, 64, 128, 128
        # concat up3 and enc1: bs, 64+64, 128, 128
        self.conv3 = DoubleConvResidBlock(64+64, 64) # bs, 64, 128, 128

        self.up4 = nn.ConvTranspose2d(64, 64, 2, 2) # bs, 64, 256, 256
        # concat up4 and enc0: bs, 64+64, 128, 128
        self.conv4 = DoubleConvResidBlock(64+64, 64) # bs, 64, 256, 256

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2), 
            nn.Conv2d(32, out_channels, 1),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        # encoder
        c1 = self.enc0(x)
        c2 = self.enc1(c1)
        c3 = self.enc2(c2)
        c4 = self.enc3(c3)
        c5 = self.enc4(c4)

        # decoder 
        u1 = self.up1(c5)
        cat1 = torch.cat([u1, c4], dim= 1)
        uc1 = self.conv1(cat1)

        u2 = self.up2(uc1)
        cat2 = torch.cat([u2, c3], dim= 1)
        uc2 = self.conv2(cat2)

        u3 = self.up3(uc2)
        cat3 = torch.cat([u3, c2], dim= 1)
        uc3 = self.conv3(cat3)

        u4 = self.up4(uc3)
        cat4 = torch.cat([u4, c1], dim= 1)
        uc4 = self.conv4(cat4)

        # return decoder
        out = self.decode(uc4)
        
        return out

    def predict(self, x, thr= 0.5):
        p = self(x)
        p[p > thr] = 255 
        p[p < thr] = 0
        return p

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step= True, 
                    on_epoch= True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        # iou = mask_intersection_over_union(y_hat.detach().cpu().numpy(), y.detach().cpu().numpy())
        self.log('val_loss', loss, on_step= True, 
                    on_epoch= True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr= self.lr)