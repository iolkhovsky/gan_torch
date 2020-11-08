import torch
import torch.nn as nn
from models.layers import ConvX1
import unittest
import numpy as np


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = ConvX1([3, 64])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvX1([64, 128])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvX1([128, 512])
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)       # 3x64x64 -> 64x64x64
        x = self.pool1(x)       # 64x64x64 -> 64x32x32
        x = self.conv2(x)       # 64x32x32 -> 128x32x32
        x = self.pool2(x)       # 128x32x32 -> 128x16x16
        x = self.conv3(x)       # 128x16x16 -> 512x16x16
        x = self.pool3(x)       # 512x16x16 -> 512x8x8
        return x


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.do = torch.nn.Dropout(p=0.8)
        self.fc1 = nn.Linear(512*8*8, 512)
        self.bn = nn.BatchNorm1d(512)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(512, 1)
        self.act_final = nn.Sigmoid()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.do(x)
        x = self.act(self.bn(self.fc1(x)))
        x = self.act_final(self.fc2(x))
        return x


class FaceDiscriminator(nn.Module):

    def __init__(self):
        super(FaceDiscriminator, self).__init__()
        self.fext = FeatureExtractor()
        self.clf = Classifier()

    def forward(self, x):
        return self.clf(self.fext(x))

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
             if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def __str__(self):
        return "FaceDiscriminator"


class TestDescriminator(unittest.TestCase):

    def test_forward_pass(self):
        batch_sz = 2
        model = FaceDiscriminator()
        test_in = torch.rand(batch_sz, 3, 64, 64)
        net_out = model.forward(test_in)
        self.assertEqual(net_out.shape, (batch_sz, 1))

    def test_back_prop(self):
        batch_sz = 2
        model = FaceDiscriminator()

        test_in = torch.rand(batch_sz, 3, 64, 64)
        out = model.forward(test_in)
        target = torch.rand(batch_sz, 1)

        loss_func = nn.BCELoss()
        loss = loss_func(out, target)
        loss.backward()
        print(loss.item())


if __name__ == "__main__":
    unittest.main()
