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
        self.path_input = os.path.join(path, 'Images')
        self.path_output = os.path.join(path, 'GroundTruth')
        self.datafile = datafile
        self.df = pd.read_csv(os.path.join(path, datafile))
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        image_input = Image.open(os.path.join(self.path_input, self.df.iloc[idx,0])).resize((224,224))
        image_input = self.transforms(image_input)
        image_output = Image.open(os.path.join(self.path_output, self.df.iloc[idx,0])).resize((224,224))
        image_output = self.transforms(image_output)

        return (image_input, image_output)
