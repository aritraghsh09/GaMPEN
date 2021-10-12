# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm

import kornia.augmentation as K

from ggt.data import FITSDataset, get_data_loader
from ggt.models import model_factory
from ggt.utils import (
    discover_devices,
    standardize_labels,
    enable_dropout,
    specify_dropout_rate,
)


def predict(
    model_path,
    dataset,
    cutout_size,
    channels,
    parallel=False,
    batch_size=256,
    n_workers=1,
    model_type="ggt",
    n_out=1,
    mc_dropout=False,
    dropout_rate=None,
):
    """Using the model defined in model path, return the output values for
    the given set of images"""

    # Discover devices
    device = discover_devices()

    # Declare the model given model_type
    cls = model_factory(model_type)
    model_args = {
        "cutout_size": cutout_size,
        "channels": channels,
        "n_out": n_out,
    }

    if model_type == "vgg16_w_stn_drp":
        model_args["dropout"] = "True"

    model = cls(**model_args)
    model = nn.DataParallel(model) if parallel else model
    model = model.to(device)

    # Changing the dropout rate if specified
    if dropout_rate is not None:
        specify_dropout_rate(model, dropout_rate)

    # Load the model
    logging.info("Loading model...")
    model.load_state_dict(torch.load(model_path))

    # Create a data loader
    loader = get_data_loader(
        dataset, batch_size=batch_size, n_workers=n_workers, shuffle=False
    )

    logging.info("Performing predictions...")
    yh = []
    model.eval()

    # Enable Monte Carlo dropout if requested
    if mc_dropout:
        logging.info("Activating Monte Carlo dropout...")
        enable_dropout(model)

    with torch.no_grad():
        for data in tqdm(loader):
            X, _ = data
            yh.append(model(X.to(device)))
    return torch.cat(yh).cpu().numpy()


@click.command()
@click.option(
    "--model_type",
    type=click.Choice(
        ["ggt", "vgg16", "ggt_no_gconv", "vgg16_w_stn", "vgg16_w_stn_drp"],
        case_sensitive=False,
    ),
    default="ggt",
)
@click.option("--model_path", type=click.Path(exists=True), required=True)
@click.option("--output_path", type=click.Path(writable=True), required=True)
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option("--cutout_size", type=int, default=167)
@click.option("--channels", type=int, default=1)
@click.option(
    "--slug",
    type=str,
    required=True,
    help="""This specifies which slug (balanced/unbalanced
              xs, sm, lg, dev) is used to perform predictions on.""",
)
@click.option("--split", type=str, required=True, default="test")
@click.option(
    "--normalize/--no-normalize",
    default=True,
    help="""The normalize argument controls whether or not, the
              loaded images will be normalized using the arcsinh function""",
)
@click.option(
    "--label_scaling",
    type=str,
    default=None,
    help="""The label scaling option controls whether to
standardize the labels or not. Set this to std for sklearn's
StandardScaling() and minmax for sklearn's MinMaxScaler().
This is especially important when predicting multiple
outputs. Note that you should pass the same argument for
label_scaling as was used during the training phase (of the
model being used for inference).""",
)
@click.option("--batch_size", type=int, default=256)
@click.option(
    "--n_workers",
    type=int,
    default=16,
    help="""The number of workers to be used during the
              data loading process.""",
)
@click.option(
    "--parallel/--no-parallel",
    default=False,
    help="""The parallel argument controls whether or not
              to use multiple GPUs when they are available""",
)
@click.option(
    "--label_cols",
    type=str,
    default="bt_g",
    help="""Enter the label column(s) separated by commas. Note
    that you should pass the exactly same argument for label_cols
    as was used during the training phase (of the model being used
    for inference). """,
)
@click.option(
    "--repeat_dims/--no-repeat_dims",
    default=False,
    help="""In case of multi-channel data, whether to repeat a two
              dimensional image as many times as the number of channels""",
)
@click.option(
    "--mc_dropout/--no-mc_dropout",
    default=False,
    help="""Turn on Monte Carlo dropout during inference.""",
)
@click.option(
    "--dropout_rate",
    type=float,
    default=None,
    help="""The dropout rate to use for all the layers in the
    model. If this is set to None, then the default dropout rate
    in the specific model is used. This option should only be
    used when you have used a non-default dropout rate during
    training and have set --mc_dropout to True. The rate should
    be set equal to the rate used during training.""",
)
@click.option(
    "--transform/--no-transform",
    default=False,
    help="""If True, the images are passed through a cropping transformation
to ensure proper cutout size""",
)
def main(
    model_path,
    output_path,
    data_dir,
    cutout_size,
    channels,
    parallel,
    slug,
    split,
    normalize,
    batch_size,
    n_workers,
    label_cols,
    model_type,
    repeat_dims,
    label_scaling,
    mc_dropout,
    dropout_rate,
    transform,
):

    # Create label cols array
    label_cols_arr = label_cols.split(",")

    # Transforming the dataset to the proper cutout size
    T = None
    if transform:
        T = nn.Sequential(
            K.CenterCrop(cutout_size),
        )

    # Load the data and create a data loader
    logging.info("Loading images to device...")
    dataset = FITSDataset(
        data_dir,
        slug=slug,
        normalize=normalize,
        split=split,
        cutout_size=cutout_size,
        channels=channels,
        label_col=label_cols_arr,
        repeat_dims=repeat_dims,
        label_scaling=label_scaling,
        transform=T if T is not None else None,
    )

    # Make predictions
    preds = predict(
        model_path,
        dataset,
        cutout_size,
        channels,
        parallel=parallel,
        batch_size=batch_size,
        n_workers=n_workers,
        model_type=model_type,
        n_out=len(label_cols_arr),
        mc_dropout=mc_dropout,
        dropout_rate=dropout_rate,
    )

    # Scale labels back to old values
    if label_scaling is not None:
        preds = standardize_labels(
            preds,
            data_dir,
            split,
            slug,
            label_cols_arr,
            label_scaling,
            invert=True,
        )

    # Write a CSV of predictions
    catalog = pd.read_csv(
        Path(data_dir) / "splits/{}-{}.csv".format(slug, split)
    )
    for i, label in enumerate(label_cols_arr):
        catalog[f"preds_{label}"] = preds[:, i]

    catalog.to_csv(output_path, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
