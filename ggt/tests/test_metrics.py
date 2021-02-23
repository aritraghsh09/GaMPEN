# -*- coding: utf-8 -*-

import torch
from ggt.metrics import ElementwiseMae

import pytest  # noqa: F401

torch.manual_seed(1)


def test_elementwise_mae():
    metric = ElementwiseMae()
    n_outputs = 5
    err = torch.arange(0, n_outputs)

    for i in range(100):
        # Fake a prediction, then add a known error
        y_pred = torch.rand(1, n_outputs)
        y = y_pred + err

        # Update the metric with this known error
        metric.update((y_pred, y))

    # Compute the metric and confirm it's equal to the error
    res = metric.compute()
    assert torch.all(torch.eq(res, err))
