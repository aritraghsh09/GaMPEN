import torch


def aleatoric_loss(outputs, targets, size_average=True):
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

    y_hat = outputs[..., 0][..., None]
    s_k = outputs[..., 1][..., None]
    y_true = targets[..., 0][..., None]

    # Compute the aleatoric loss
    first_term = 0.5 * torch.pow(y_hat - y_true, 2) * torch.exp(-1.0 * s_k)
    second_term = 0.5 * s_k
    aleatoric_loss = first_term + second_term

    if size_average:
        aleatoric_loss = torch.mean(aleatoric_loss)
    else:
        aleatoric_loss = torch.sum(aleatoric_loss)

    return aleatoric_loss
