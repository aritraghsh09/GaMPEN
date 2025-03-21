import torch.nn as nn

from torchvision import models


def vgg16(cutout_size, channels, n_out=1, pretrained=True):

    if pretrained:
        model = models.vgg16(weights="VGG16_Weights.DEFAULT")
    else:
        model = models.vgg16()

    model.classifier[6] = nn.Linear(4096, n_out)

    return model
