import torch


def aleatoric_cov_loss(outputs, targets, num_var=3, average=True):
    """
    Computes the Aleatoric Loss while including the full
    covariance matrix of the outputs.

    If you are predicting for n output variables, then the 
    number of output neuros required for this loss is
    (3n + n^2)/2.

    Args:
        outputs: (tensor) - predicted outputs from the model
        targets: (tensor) - ground truth labels
        size_average: (bool) - if True, the losses are
               averaged over all elements of the batch
    Returns:
        aleatoric_cov_loss: (tensor) - aleatoric loss

    Formula:
        loss = 0.5 * [y_i - y_i_hat].T * cov_mat_inv 
                * [y_i - y_i_hat] + 0.5 * log(det(cov_mat))
    """

    # Checking that all dimensions match up properly
    num_out = outputs.shape[len(outputs.shape) - 1]
    if num_out != (3*num_var + num_var**2)/2:
        raise ValueError(
            "The number of predicted variables should be equal to "
            "3n + n^2/2 for calculation of aleatoric loss"
        )
    batch_size = outputs.shape[0]

    y_hat = outputs[..., :int(num_var)]
    var = outputs[..., int(num_var):int(num_var*2)]
    covs = outputs[..., int(num_var*2):]


    #Calculating the Covariance matrix
    #Here we write the Covirance matrix as LDLT decomposition
    #Cov. Mat = LDLT
    #And L = (I + N)

    D = torch.diag_embed(var)
    N = torch.zeros(batch_size, num_var, num_var)

    i,j = torch.tril_indices(num_var, num_var, -1)
    N[:,i,j] = covs

    I = torch.eye(num_var)
    I = I.reshape(1, num_var, num_var)
    I = I.repeat(batch_size, 1, 1)
    #L = I + N

    #LT = torch.transpose(L, 1, 2)

    #cov_mat = torch.chain_matmul(L, D, LT)
    #cov_mat = torch.bmm(torch.bmm(L, D), LT)

    Nk = torch.zeros_like(N)
    for k in range(1, num_var):
        Nk = Nk + (-1)**k * torch.matrix_power(N,k)
    L_inv = I + Nk
    L_inv_T = torch.transpose(L_inv, 1, 2)

    D_inv_ele = 1.0 / torch.diagonal(D,0,1,2)
    D_inv = torch.diag_embed(D_inv_ele)

    cov_mat_inv = torch.bmm(torch.bmm(L_inv_T, D_inv), L_inv)

    log_det_cov = torch.sum(torch.diagonal(D,0,1,2).log(), 1)

    diff = y_hat - targets
    diff = torch.unsqueeze(diff,-1)
    diffT = torch.transpose(diff, 1, 2)

    #Compute the aleatoric loss
    aleatoric_loss = (
        0.5 * torch.bmm(torch.bmm(diffT, cov_mat_inv), diff).squeeze(dim=1).squeeze(dim=1) + 0.5 * log_det_cov
    )

    if average:
        aleatoric_loss = torch.mean(aleatoric_loss)
    else:
        aleatoric_loss = torch.sum(aleatoric_loss)

    return aleatoric_loss