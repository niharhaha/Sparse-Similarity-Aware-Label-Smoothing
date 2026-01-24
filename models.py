import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from tqdm import tqdm
from dataset_utils import get_data_loaders
import pandas as pd
import timm

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(64*7*7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
def CifarResNet18(num_classes):
    model = models.resnet18(weights = None)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def CifarDenseNet121(num_classes):
    model = timm.create_model(
        "densenet121",
        pretrained=False,
        num_classes=num_classes
    )
    return model

def TinyResNet34(num_classes):
    model = timm.create_model(
        "resnet34",
        pretrained=False,
        num_classes=num_classes
    )
    model.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = torch.nn.Identity()
    return model


def TinyDenseNet121(num_classes):
    return timm.create_model(
        "densenet121",
        pretrained=False,
        num_classes=num_classes,
        img_size=64
    )
