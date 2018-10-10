import torch
from torch.optim import Adam
import os
from DataLoaderClass import DataLoaderClass
from Unet import UNet
from torchvision import transforms
from trainer import trainer

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

batch_size = int(input('Batch Size: '))
model = UNet(3, depth=4, merge_mode='concat').cuda()
optimizer = Adam(model.parameters(), lr = 0.0002)

dataset =  DataLoaderClass('/home/sharath/Downloads/MSRC_ObjCategImageDatabase_v2/', 'data.csv', transforms = transforms)

data_loader = torch.utils.data.DataLoader(dataset, batch_size)

trainer(model, optimizer, data_loader,100)
