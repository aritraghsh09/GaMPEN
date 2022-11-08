import torch


def aleatoric_loss(outputs, targets, average=True):
    """
    Computes the aleatoric loss.
    Args:
        outputs: (tensor) - predicted outputs from the model
        targets: (tensor) - ground truth labels
        size_average: (bool) - if True, the losses are
               averaged over all elements of the batch
    Returns:
        aleatoric_loss: (tensor) - aleatoric loss
    """

    num_out = outputs.shape[len(outputs.shape) - 1]
    if num_out % 2 != 0:
        raise ValueError(
            "The number of predicted variables should be divisible by "
            "2 for calculation of aleatoric loss"
        )

    y_hat = outputs[..., : int(num_out / 2)]
    s_k = outputs[..., -int(num_out / 2) :]

    # Compute the aleatoric loss
    aleatoric_loss = (
        0.5 * torch.pow(y_hat - targets, 2) * torch.exp(-1.0 * s_k) + 0.5 * s_k
    )

    if average:
        aleatoric_loss = torch.mean(aleatoric_loss)
    else:
        aleatoric_loss = torch.sum(aleatoric_loss)

    return aleatoric_loss
