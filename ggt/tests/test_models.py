# -*- coding: utf-8 -*-

import torch
from ggt.models import model_factory

import pytest  # noqa: F401

torch.manual_seed(1)


def test_ggt():
    cls = model_factory("ggt")
    model = cls(169, 1, 1)
    print(model)


def test_ggt_no_gconv():
    cls = model_factory("ggt_no_gconv")
    model = cls(169, 1, 1)
    print(model)


def test_vgg16():
    cls = model_factory("vgg16")
    model = cls(169, 1, 1, pretrained=False)
    print(model)


def test_vgg16_w_stn():
    cls = model_factory("vgg16_w_stn")
    model = cls(169, 1, 1, pretrained=False)
    print(model)


def test_vgg16_w_stn_drp():
    cls = model_factory("vgg16_w_stn_drp")
    model = cls(169, 1, 1, dropout=True, pretrained=False)
    print(model)


@pytest.mark.skip(reason="requires network connection")
def test_vgg16_pretrained():
    cls = model_factory("vgg16")
    model = cls(169, 1, 1, pretrained=True)
    print(model)


@pytest.mark.skip(reason="requires network connection")
def test_vgg16_w_stn_drp_pretrained():
    cls = model_factory("vgg16_w_stn_drp")
    model = cls(169, 1, 1, dropout=True, pretrained=True)
    print(model)
