def aleatoric_cov_loss_2(outputs, targets, average=True):
    '''Calculates the Aleatoric Loss
    while accounting for the full covariance matrix'''

    # get the mean and covariance matrix
    mean, cov = aleatoric_cov_2(outputs)
    # get the number of outputs
    n_outputs = outputs.size(1)
    # get the number of targets
    n_targets = targets.size(1)
    # get the number of samples
    n_samples = outputs.size(0)
    # get the number of dimensions
    n_dim = outputs.size(2)

    # calculate the loss
    loss = 0
    for i in range(n_outputs):
        for j in range(n_targets):
            # get the target
            target = targets[:, j]
            # get the output
            output = outputs[:, i]
            # calculate the loss
            loss += torch.sum(torch.mm(cov[i, j], (target - output).view(n_samples, 1)))

    # average the loss
    if average:
        loss = loss / (n_outputs * n_targets)

    return loss

def aleatoric_cov_2(outputs):
    '''Calculates the mean and covariance matrix'''

    # get the number of outputs
    n_outputs = outputs.size(1)
    # get the number of samples
    n_samples = outputs.size(0)
    # get the number of dimensions
    n_dim = outputs.size(2)

    # calculate the mean
    mean = torch.mean(outputs, dim=0)

    # calculate the covariance matrix
    cov = torch.zeros(n_outputs, n_outputs)
    for i in range(n_outputs):
        for j in range(n_outputs):
            cov[i, j] = torch.mean(torch.pow(outputs[:, i] - mean[i], 2) * torch.pow(outputs[:, j] - mean[j], 2))

    return mean, cov 

 