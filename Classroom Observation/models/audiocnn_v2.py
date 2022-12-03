import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class U8KCNN(nn.Module):
    def __init__(self,num_classses, drop_rate = 0.0):
        super(U8KCNN, self).__init__()
        channels = 1
        categories = 5
        
        self.conv1 = nn.Conv2d(channels, 128, 9, padding = "same" )
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.tanh1 = nn.Tanh()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(128, 96, 5, padding = "same")
        self.batchnorm2 = nn.BatchNorm2d(96)
        self.tanh2 = nn.Tanh()
        self.conv3 = nn.Conv2d(96, 96, 5, padding = "same")
        self.batchnorm3 = nn.BatchNorm2d(96)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        self.dropout1 = nn.Dropout(p = drop_rate)
        self.conv4 = nn.Conv2d(96, 32, 11, padding = "same")
        self.batchnorm4 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(p = drop_rate)
        self.conv5 = nn.Conv2d(32, 64, 5, padding = "same")
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2)
        self.dropout3 = nn.Dropout(p = drop_rate)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features = 128, out_features = 1024)
        self.dense2= nn.Linear(in_features = 1024, out_features = categories)


    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.tanh1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.tanh2(out)
        out = self.conv3(out)
        out = self.batchnorm3(out)
        out = self.maxpool2(out)
        out = self.dropout1(out)
        out = self.conv4(out)
        out = self.batchnorm4(out)
        out = self.dropout2(out)
        out = self.conv5(out)
        out = self.maxpool3(out)
        out = self.dropout3(out)
        out = self.flatten(out)
        out = self.dense1(out)
        out = self.dense2(out)
        return out


def build_U8KCNN(num_classes):
    logger.info(f"Model: URBANK-8k v2 CNN (num_classes) = {num_classes}")
    return U8KCNN(
        num_classses= num_classes,
        drop_rate=0.1
    )

