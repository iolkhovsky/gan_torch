import torch.nn as nn
from torch.nn.init import kaiming_normal_ as he_normal


class ConvX2(nn.Module):

    def __init__(self, channels):
        super(ConvX2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.act2 = nn.ReLU()
        he_normal(self.conv1.weight)
        he_normal(self.conv2.weight)
        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(self.bn1(x))
        x = self.conv2(x)
        x = self.act2(self.bn2(x))
        return x


class ConvX1(nn.Module):

    def __init__(self, channels):
        super(ConvX1, self).__init__()
        self.conv = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels[1])
        self.act = nn.LeakyReLU(0.2)
        # he_normal(self.conv.weight)
        nn.init.normal_(self.conv.weight.data, 0.0, 0.02)
        nn.init.normal_(self.bn.weight.data, 1.0, 0.02)
        nn.init.constant_(self.bn.bias.data, 0)
        return

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class UpSample(nn.Module):

    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1, bias=False)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(channels[1])
        nn.init.normal_(self.conv.weight.data, 0.0, 0.02)
        nn.init.normal_(self.bn.weight.data, 1.0, 0.02)
        nn.init.constant_(self.bn.bias.data, 0)
        return

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
