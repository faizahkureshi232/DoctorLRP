import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

class BT_VGGModel(nn.Module):
    def __init__(self):
        super(BT_VGGModel, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)

        # replace output layer according to problem
        in_feats = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(in_feats, 4)

    def forward(self, x):
        x = self.vgg16(x)
        return x