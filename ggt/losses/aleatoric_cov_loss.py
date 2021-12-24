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
        loss = 0.5 * [Y - Y_hat].T * cov_mat_inv
                * [Y - Y_hat] + 0.5 * log(det(cov_mat))
    """

    # Checking that all dimensions match up properly
    num_out = outputs.shape[len(outputs.shape) - 1]
    if num_out != (3 * num_var + num_var ** 2) / 2:
        raise ValueError(
            "The number of predicted variables should be equal to "
            "3n + n^2/2 for calculation of aleatoric loss"
        )
    batch_size = outputs.shape[0]

    # Taking the inverse of a tensor and calculating
    # it's determinant are usually not recommended operations
    # due to spped and numerical stability. Thus we use
    # LDL decomposition to calculate the inverse and determinant
    # of the covariance matrix. We use the values of the outputs
    # of the neurons as the entries of the L and D matrices.
    # cov_mat = L * D * L.T
    # where D is a diagonal matrix and L is a lower triangular
    # matrix with all diagonal elements set to 1.
    # cov_mat_inv = (L^-1).T * D^-1 * (L^-1)
    y_hat = outputs[..., : int(num_var)]
    var = outputs[..., int(num_var) : int(num_var * 2)]
    covs = outputs[..., int(num_var * 2) :]

    D = torch.diag_embed(var)
    D = D.to(outputs.device)

    # We further write L = I + N for ease of calculation
    # L^-1 = (I+N)^-1 = I + Sum_j (-1)**j * N^j
    # I is written as Id in code
    N = torch.zeros(batch_size, num_var, num_var)
    N = N.to(outputs.device)

    i, j = torch.tril_indices(num_var, num_var, -1)
    N[:, i, j] = covs

    Id = torch.eye(num_var).to(outputs.device)
    Id = Id.reshape(1, num_var, num_var)
    Id = Id.repeat(batch_size, 1, 1)

    # Extra Code for quick computation of some matrices
    # L = Id + N
    # LT = torch.transpose(L, 1, 2)
    # cov_mat = torch.bmm(torch.bmm(L, D), LT)

    Nk = torch.zeros_like(N).to(outputs.device)
    for k in range(1, num_var):
        Nk = Nk + (-1) ** k * torch.matrix_power(N, k)
    L_inv = Id + Nk
    L_inv_T = torch.transpose(L_inv, 1, 2)

    # Becasue D is a diagonal matrix,
    # D^-1 = 1/D_ii for all i
    D_inv_ele = 1.0 / torch.diagonal(D, 0, 1, 2)
    D_inv = torch.diag_embed(D_inv_ele).to(outputs.device)

    cov_mat_inv = torch.bmm(torch.bmm(L_inv_T, D_inv), L_inv)

    # log(det(cov_mat)) = log(det(LDL.T)) = log(\Pi_i D_ii)
    #                                     = Sum_i log(D_ii)
    log_det_cov = torch.sum(torch.diagonal(D, 0, 1, 2).log(), 1)

    diff = targets - y_hat
    # unsqueezing just so that the difference matrix can be
    # transposed
    diff = torch.unsqueeze(diff, -1)
    diffT = torch.transpose(diff, 1, 2)

    # Compute the aleatoric loss. The squeeze command is used
    # to ensure that the result of the matrix multiplication comes
    # out with the proper dimensions.
    aleatoric_loss = (
        0.5
        * torch.bmm(torch.bmm(diffT, cov_mat_inv), diff)
        .squeeze(dim=1)
        .squeeze(dim=1)
        + 0.5 * log_det_cov
    )

    # Averaging or summing over the batch
    if average:
        aleatoric_loss = torch.mean(aleatoric_loss)
    else:
        aleatoric_loss = torch.sum(aleatoric_loss)

    return aleatoric_loss
