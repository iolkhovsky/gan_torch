import torch
import torch.nn as nn
from torch.nn.functional import max_pool2d
from models.layers import ConvX2, UpSample
import unittest
import numpy as np


class FaceGenerator(nn.Module):

    def __init__(self):
        super(FaceGenerator, self).__init__()
        self.fc1 = nn.Linear(64, 4*4*512, bias=False)
        self.act = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm1d(4*4*512)
        self.up1 = UpSample([512, 256])
        self.up2 = UpSample([256, 128])
        self.up3 = UpSample([128, 64])
        self.up4 = UpSample([64, 3])
        nn.init.normal_(self.fc1.weight.data, 0.0, 0.02)
        nn.init.normal_(self.bn.weight.data, 1.0, 0.02)
        nn.init.constant_(self.bn.bias.data, 0)
        return

    def forward(self, x):
        x = self.bn(self.fc1(x))             # 64 -> 4*4*512
        x = self.act(x)
        x = x.view(-1, 512, 4, 4)       # 4*4*512 -> 512x4x4
        x = self.up1(x)             # 512x4x4 -> 256x8x8
        x = self.up2(x)             # 256x8x8 -> 128x16x16
        x = self.up3(x)             # 128x16x16 -> 64x32x32
        x = self.up4(x)             # 64x32x32 -> 3x64x64
        return x

    def __str__(self):
        return "FaceGenerator"


class TestGenerator(unittest.TestCase):

    def test_forward_pass(self):
        batch_sz = 2
        model = FaceGenerator()
        test_in = torch.rand(batch_sz, 64)
        net_out = model.forward(test_in)
        self.assertEqual(net_out.shape, (batch_sz, 3, 64, 64))
        return


if __name__ == "__main__":
    unittest.main()
