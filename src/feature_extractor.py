from turtle import forward
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any
from src import config
from src.device import device
from typing import OrderedDict


class ResnetFPNFeactureExtractor(nn.Module):
    def __init__(self):
        super(ResnetFPNFeactureExtractor, self).__init__()
        self.layers = list(models.resnet50(pretrained=True).children())
        self.conv2 = nn.Sequential(*self.layers[0:5])
        self.conv3 = nn.Sequential(*self.layers[5:6])
        self.conv4 = nn.Sequential(*self.layers[6:7])
        self.conv5 = nn.Sequential(*self.layers[7:8])

        self.lateral_conv5 = nn.Conv2d(2048, config.fpn_feat_channels, 1, 1)
        self.lateral_conv4 = nn.Conv2d(1024, config.fpn_feat_channels, 1, 1)
        self.lateral_conv3 = nn.Conv2d(512, config.fpn_feat_channels, 1, 1)
        self.lateral_conv2 = nn.Conv2d(256, config.fpn_feat_channels, 1, 1)

        self.upscale = lambda input: F.interpolate(input, scale_factor=2)
        self.freeze_params()
        
    def freeze_params(self):
        modules = [
            # self.conv2.children(), 
            # self.conv3.children(), 
            # self.conv4.children(), 
            # self.conv5.children()
        ]
        for module in modules:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    for param in layer.parameters():
                        param.requires_grad = False 

    def forward(self, x):
        c2 = self.conv2(x)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        p5 = self.lateral_conv5(c5)
        p4 = self.lateral_conv4(c4) + self.upscale(p5)
        p3 = self.lateral_conv3(c3) + self.upscale(p4)
        p2 = self.lateral_conv2(c2) + self.upscale(p3)

        return [p2, p3, p4, p5]