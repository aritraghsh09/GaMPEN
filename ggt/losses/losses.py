import torch.nn as nn

from .aleatoric_loss import aleatoric_loss
from .aleatoric_cov_loss import aleatoric_cov_loss


class AleatoricLoss(nn.Module):
    def __init__(self, average=True):
        super(AleatoricLoss, self).__init__()
        self.average = average

    def forward(self, outputs, targets):
        return aleatoric_loss(outputs, targets, average=self.average)


class AleatoricCovLoss(nn.Module):
    def __init__(self, average=True):
        super(AleatoricCovLoss, self).__init__()
        self.average = average

    def forward(self, outputs, targets):
        return aleatoric_cov_loss(outputs, targets, average=self.average)
