# -*- coding: utf-8 -*-

import torch
from ggt.models import model_factory

import pytest  # noqa: F401

torch.manual_seed(1)


def test_ggt():
    cls = model_factory("ggt")
    model = cls(169, 1, 1, dropout=0.0)
    print(model)


def test_ggt_dropout():
    cls = model_factory("ggt")
    model = cls(169, 1, 1, dropout=0.5)
    print(model)


def test_ggt_no_gconv():
    cls = model_factory("ggt_no_gconv")
    model = cls(169, 1, 1, dropout=0.0)
    print(model)


def test_ggt_no_gconv_dropout():
    cls = model_factory("ggt_no_gconv")
    model = cls(169, 1, 1, dropout=0.5)
    print(model)


def test_vgg():
    cls = model_factory("vgg")
    model = cls(
        169, 1, 1, dropout=0.0, pretrained=False, use_spatial_transformer=False
    )
    print(model)


def test_vgg_dropout():
    cls = model_factory("vgg")
    model = cls(
        169, 1, 1, dropout=0.5, pretrained=False, use_spatial_transformer=False
    )
    print(model)


def test_vgg_stn():
    cls = model_factory("vgg")
    model = cls(
        169, 1, 1, dropout=0.0, pretrained=False, use_spatial_transformer=True
    )
    print(model)


def test_vgg_dropout_stn():
    cls = model_factory("vgg")
    model = cls(
        169, 1, 1, dropout=0.5, pretrained=False, use_spatial_transformer=True
    )
    print(model)


@pytest.mark.skip(reason="requires network connection")
def test_vgg_pretrained():
    cls = model_factory("vgg")
    model = cls(
        169, 1, 1, dropout=0.0, pretrained=True, use_spatial_transformer=False
    )
    print(model)


@pytest.mark.skip(reason="requires network connection")
def test_vgg_dropout_pretrained():
    cls = model_factory("vgg")
    model = cls(
        169, 1, 1, dropout=0.5, pretrained=True, use_spatial_transformer=False
    )
    print(model)


@pytest.mark.skip(reason="requires network connection")
def test_vgg_stn_pretrained():
    cls = model_factory("vgg")
    model = cls(
        169, 1, 1, dropout=0.0, pretrained=True, use_spatial_transformer=True
    )
    print(model)


@pytest.mark.skip(reason="requires network connection")
def test_vgg_dropout_stn_pretrained():
    cls = model_factory("vgg")
    model = cls(
        169, 1, 1, dropout=0.5, pretrained=True, use_spatial_transformer=True
    )
    print(model)
