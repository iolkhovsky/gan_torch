import torch
import torch.nn as nn
from torch.nn.functional import max_pool2d
from models.layers import ConvX2, ConvX1
import unittest
import numpy as np


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = ConvX1([3, 64])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvX1([64, 128])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvX1([128, 256])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvX1([256, 512])
        self.pool4 = nn.MaxPool2d(2)
        return

    def forward(self, x):
        x = self.conv1(x)       # 3x64x64 -> 64x64x64
        x = self.pool1(x)       # 64x64x64 -> 64x32x32
        x = self.conv2(x)       # 64x32x32 -> 128x32x32
        x = self.pool2(x)       # 128x32x32 -> 128x16x16
        x = self.conv3(x)       # 128x16x16 -> 256x16x16
        x = self.pool3(x)       # 256x16x16 -> 256x8x8
        x = self.conv4(x)       # 256x8x8 -> 512x8x8
        x = self.pool4(x)       # 512x8x8 -> 512x4x4
        return x


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.do = torch.nn.Dropout(p=0.8)
        self.fc = nn.Linear(128*4*4, 1)
        nn.init.normal_(self.fc.weight.data, 0.0, 0.02)
        return

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.do(x)
        x = self.fc(x)
        return x


class FaceDiscriminator(nn.Module):

    def __init__(self):
        super(FaceDiscriminator, self).__init__()
        self.fext = FeatureExtractor()
        self.clf = Classifier()
        self.act = nn.Sigmoid()
        return

    def forward(self, x):
        x = self.fext(x)
        logits = self.clf(x)
        probs = self.act(logits)
        return probs

    def __str__(self):
        return "FaceDiscriminator"


class TestDescriminator(unittest.TestCase):

    def test_forward_pass(self):
        batch_sz = 2
        model = FaceDiscriminator()
        test_in = torch.from_numpy(np.arange(64*64*3*batch_sz).reshape(batch_sz, 3, 64, 64).astype(np.float32))
        net_out = model.forward(test_in)
        self.assertEqual(net_out.shape, (batch_sz, 1))
        return

    def test_back_prop(self):
        batch_sz = 2
        model = FaceDiscriminator()

        input = torch.from_numpy(np.arange(64*64*3*batch_sz).reshape(batch_sz, 3, 64, 64).astype(np.float32))
        out = model.forward(input)
        target = torch.rand(batch_sz, 1)

        loss_func = nn.BCELoss()
        loss = loss_func(out, target)
        loss.backward()
        return


if __name__ == "__main__":
    unittest.main()
