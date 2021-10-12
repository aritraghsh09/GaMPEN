import torch.nn as nn

from .aleatoric_loss import aleatoric_loss


class AleatoricLoss(nn.Module):
    def __init__(self, size_average=True):
        super(AleatoricLoss, self).__init__()
        self.size_average = size_average

    def forward(self, outputs, targets):
        return aleatoric_loss(outputs, targets, size_average=self.size_average)
