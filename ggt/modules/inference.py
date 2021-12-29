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

    if "drp" in model_type.split("_"):
        logging.info(
            "Using dropout rate of {} in the model".format(dropout_rate)
        )
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
    return torch.cat(yh)


@click.command()
@click.option(
    "--model_type",
    type=click.Choice(
        [
            "ggt",
            "vgg16",
            "ggt_no_gconv",
            "vgg16_w_stn",
            "vgg16_w_stn_drp",
            "vgg16_w_stn_at_drp",
            "vgg16_w_stn_oc_drp",
        ],
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
    "--n_runs",
    type=int,
    default=1,
    help="""The number of times to run inference. This is helpful
    when usng mc_dropout""",
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
@click.option(
    "--errors/--no-errors",
    default=False,
    help="""If True and if the model allows for it, aleatoric uncertainties are
    written to the output file. Only set this to True if you trained the model
    with aleatoric loss.""",
)
@click.option(
    "--cov_errors/--no-cov_errors",
    default=False,
    help="""If True and if the model allows for it, aleatoric uncertainties are
    written to the output file. Only set this to True if you trained the model
    with aleatoric_cov loss.""",
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
    errors,
    cov_errors,
    n_runs,
):

    # Create label cols array
    label_cols_arr = label_cols.split(",")

    # Calculating the number of outputs
    if errors:
        n_out = int(len(label_cols_arr) * 2)
    elif cov_errors:
        n_var = len(label_cols_arr)
        n_out = int((3 * n_var + n_var ** 2) / 2)
    else:
        n_out = len(label_cols_arr)

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

    for run_num in range(1, n_runs + 1):

        logging.info(f"Running inference run {run_num}")

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
            n_out=n_out,
            mc_dropout=mc_dropout,
            dropout_rate=dropout_rate,
        )

        if errors:
            # Note that here we are drawing the
            # predictions from a distribution
            # in the transformed/scaled label space.
            means = preds[..., : int(n_out / 2)]
            sks = preds[..., -int(n_out / 2) :]
            sigmas = torch.sqrt(torch.exp(sks))

            # preds = np.random.normal(means, sigmas)
            preds = torch.distributions.Normal(means, sigmas).sample()
            preds = preds.cpu().numpy()
            means = means.cpu().numpy()
            sigmas = sigmas.cpu().numpy()

        elif cov_errors:
            # Note that here we are drawing the
            # predictions from a distribution
            # in the transformed/scaled label space.
            num_var = len(label_cols_arr)

            y_hat = preds[..., : int(num_var)]
            var = torch.exp(preds[..., int(num_var) : int(num_var * 2)])
            covs = preds[..., int(num_var * 2) :]

            D = torch.diag_embed(var)
            D = D.to(preds.device)

            N = torch.zeros(preds.shape[0], num_var, num_var)
            N = N.to(preds.device)

            i, j = torch.tril_indices(num_var, num_var, -1)
            N[:, i, j] = covs

            Id = torch.eye(num_var).to(preds.device)
            Id = Id.reshape(1, num_var, num_var)
            Id = Id.repeat(preds.shape[0], 1, 1)

            L = Id + N
            LT = torch.transpose(L, 1, 2)
            cov_mat = torch.bmm(torch.bmm(L, D), LT)

            preds = torch.distributions.MultivariateNormal(
                y_hat, cov_mat
            ).sample()
            preds = preds.cpu().numpy()

        else:
            preds = preds.cpu().numpy()

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

            if errors:
                catalog[f"transformd_mean_{label}"] = means[:, i]
                catalog[f"transformed_sigma_{label}"] = sigmas[:, i]

            catalog[f"preds_{label}"] = preds[:, i]

        catalog.to_csv(output_path + f"inf_{run_num}.csv", index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
