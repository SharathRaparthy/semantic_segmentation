import numpy as np
from PIL import Image
import torch
import pandas as pd
from torch.utils import data
from torchvision import transforms
import os

class DataLoaderClass(data.Dataset):
    """docstring for ."""
    def __init__(self, path, datafile, transforms):
        super(DataLoaderClass, self).__init__()
        self.transforms = transforms
        self.path = path
        self.path_input = os.path.join(path, 'dataset_1')
        self.datafile = datafile
        self.df = pd.read_csv(os.path.join(path, datafile))
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        image_input = Image.open(os.path.join(self.path_input, self.df.iloc[idx,0]))
        image_input = self.transforms(image_input)

        return image_input
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5, 0.5, 0.5))])
data = DataLoaderClass('/home/sharath/gym_duckietown','/home/sharath/semantic_segmentation/data.csv',transforms = transforms)
data = torch.utils.data.DataLoader(data, batch_size = 64)

for i in data:
    print(i)
