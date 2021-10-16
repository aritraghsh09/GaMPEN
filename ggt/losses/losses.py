import torch.nn as nn

from .aleatoric_loss import aleatoric_loss


class AleatoricLoss(nn.Module):
    def __init__(self, average=True):
        super(AleatoricLoss, self).__init__()
        self.average = average

    def forward(self, outputs, targets):
        return aleatoric_loss(outputs, targets, average=self.average)