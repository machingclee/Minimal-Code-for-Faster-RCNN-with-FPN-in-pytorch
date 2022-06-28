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


class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        self.vgg = models.vgg16(pretrained=True).to(device)
        self.features = self.vgg.features
        self.out_channels = None
        self.feature_extraction = nn.Sequential(*self._get_layers())
        self.freeze_vgg_bottom_layers()

    def freeze_vgg_bottom_layers(self):
        for layer in list(self.feature_extraction.children())[0:9]:
            if isinstance(layer, nn.Conv2d):
                for param in layer.parameters():
                    param.requires_grad = False

    def vgg_weight_init_upper_layers(self):
        for layer in list(self.feature_extraction.children())[9:]:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def unfreeze_vgg(self):
        for param in self.vgg.parameters():
            param.requires_grad = True

    def _get_layers(self):
        dummy_img = torch.randn((1, 3, config.input_height, config.input_width)).to(device)
        x = dummy_img
        desired_layers = []
        for feat in self.features:
            x = feat(x)
            if x.shape[2] < config.input_height // 16:
                # desired ouput shape is 1024//16 = 64
                break
            desired_layers.append(feat)
            self.out_channels = x.shape[1]
        return desired_layers

    def forward(self, x):
        return self.feature_extraction(x)


if __name__ == "__main__":
    renset = models.resnet50(pretrained=True)
    x = torch.randn((1, 3, 800, 800))
    ResnetFPNFeactureExtractor()(x)
