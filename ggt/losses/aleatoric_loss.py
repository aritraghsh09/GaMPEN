import torch
import torch.nn as nn


def aleatoric_loss(outputs, targets, average=True):
    """
    Computes the aleatoric loss.
    Args:
        outputs: (tensor) - predicted outputs from the model
        targets: (tensor) - ground truth labels
        size_average: (bool) - if True, the losses are
               averaged over all elements of the batch
    Returns:
        loss: (tensor) - aleatoric loss
    """

    if outputs.shape[-1] % 2 != 0:
        raise ValueError(
            "The number of predicted variables should be divisible by "
            "2 for calculation of aleatoric loss"
        )

    k = outputs.shape[-1] // 2
    yh, sk = outputs[..., :k], outputs[..., -k:]

    # Compute the aleatoric loss
    loss = 0.5 * torch.pow(yh - targets, 2) * torch.exp(-1.0 * sk) + 0.5 * sk

    if average:
        loss = torch.mean(loss)
    else:
        loss = torch.sum(loss)

    return loss


class AleatoricLoss(nn.Module):
    def __init__(self, average=True):
        super(AleatoricLoss, self).__init__()
        self.average = average

    def forward(self, outputs, targets):
        return aleatoric_loss(outputs, targets, average=self.average)
