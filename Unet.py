import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
def conv3x3(input_channels, output_channels):
    return nn.Conv2d(input_channels,output_channels,stride = 1, padding = 1, kernel_size = 3)
def conv1x1(input_channels, output_channels):
    return nn.Conv2d(input_channels,output_channels,stride = 1, kernel_size = 1)


def up_conv(input_channels,output_channels,mode = "transpose"):
    if mode == "transpose":
        return nn.Sequential(nn.Upsample(input_channels,output_channels,mode = transpose))
    else:
        return nn.Sequential(nn.Upsample(scale_factor= 2,mode = 'bilinear'))

class DownNet(nn.Module):
    """docstring for UNet."""
    def __init__(self, input_channels, output_channels,pooling = True):
        super(DownNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.padding = padding
        self.conv1 = nn.Sequential(conv3x3(self.input_channels,self.output_channels), nn.BatchNorm2d(output_channels))
        self.conv2 = nn.Sequential(conv3x3(input_channels,output_channels), nn.BatchNorm2d(output_channels))
        if self.padding:
            self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    def forward(self, x):
        x = F.ReLU(self.conv1(x))
        x = F.ReLU(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool
