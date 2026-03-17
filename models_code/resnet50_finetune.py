import torch
import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes=2, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model
