import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (3,3), padding= 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, (3,3), padding= 1),
                nn.BatchNorm2d(out_channels)
        )
        self.mapping = nn.Conv2d(in_channels, out_channels, (1,1))
        
    def forward(self, x):
        out = self.block(x)
        out = out + self.mapping(x)
        out = F.relu(out)
        return out

class Unet(pl.LightningModule):

    def __init__(self, in_channels, out_channels, 
                lr=1e-3):
        super(Unet, self).__init__()
        
        self.lr = lr

        self.conv1 = DoubleConv(in_channels, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        self.conv5 = DoubleConv(512, 1024)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # halves the H X W
        
        self.up1 = nn.ConvTranspose2d(in_channels = 1024, 
                    out_channels =512, kernel_size = 2, stride= 2)
        self.up2 = nn.ConvTranspose2d(in_channels = 512, 
                    out_channels =256, kernel_size = 2, stride= 2)
        self.up3 = nn.ConvTranspose2d(in_channels = 256, 
                    out_channels =128, kernel_size = 2, stride= 2)
        self.up4 = nn.ConvTranspose2d(in_channels = 128, 
                    out_channels =64, kernel_size = 2, stride= 2)
        
        self.up_conv1 = DoubleConv(1024, 512)
        self.up_conv2 = DoubleConv(512, 256)
        self.up_conv3 = DoubleConv(256, 128)
        self.up_conv4 = DoubleConv(128, 64)

        self.decode = nn.Conv2d(64, out_channels, 1)

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss, on_step= True, 
                    on_epoch= True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('test_loss', loss)
        return loss 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr= self.lr)

    