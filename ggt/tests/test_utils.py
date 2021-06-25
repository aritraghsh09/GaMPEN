# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from ggt.utils import specify_dropout_rate

import pytest  # noqa: F401

torch.manual_seed(1)

def test_specify_dropout_rate():

    # Defining a basic model
    model = nn.Sequential(
       nn.Dropout(0.5),
       nn.Linear(2048, 1024),
       nn.ReLU(inplace=True),
       nn.Dropout(0.5),
       nn.Linear(1024,512),
       nn.ReLU(inplace=True),
       nn.Linear(512,1)
    )

    # Picking a random dropout rate to set
    dropout_rate = torch.rand(1)

    # Setting the dropout rates
    specify_dropout_rate(model,float(dropout_rate))

    # Storing the Dropout rates from the model in an array
    model_dropout_rates = []
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            model_dropout_rates.append(m.p)

    assert torch.all(torch.eq(dropout_rate, torch.tensor(model_dropout_rates)))


