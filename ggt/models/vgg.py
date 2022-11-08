import torch.nn as nn

from torchvision import models


def vgg16(cutout_size, channels, n_out=1, pretrained=True):

    model = models.vgg16(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, n_out)

    return model
