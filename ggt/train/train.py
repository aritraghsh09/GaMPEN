# -*- coding: utf-8 -*-
import click
import logging
import math
from pathlib import Path
from functools import partial

import mlflow

import torch
import torch.nn as nn
import torch.optim as opt

import kornia.augmentation as K

from ggt.data import FITSDataset, get_data_loader
from ggt.models import model_factory, model_stats, save_trained_model
from ggt.train import create_trainer
from ggt.utils import discover_devices, specify_dropout_rate
from ggt.visualization.spatial_transform import visualize_spatial_transform
from ggt.losses import AleatoricLoss, AleatoricCovLoss


@click.command()
@click.option("--experiment_name", type=str, default="demo")
@click.option(
    "--run_id",
    type=str,
    default=None,
    help="""The run id. Practically this only needs to be used
if you are resuming a previosuly run experiment""",
)
@click.option(
    "--run_name",
    type=str,
    default=None,
    help="""A run is supposed to be a sub-class of an experiment.
So this variable should be specified accordingly""",
)
@click.option(
    "--model_type",
    type=click.Choice(
        [
            "ggt",
            "vgg16",
            "ggt_no_gconv",
            "vgg16_w_stn",
            "vgg16_w_stn_drp",
            "vgg16_w_stn_drp_2",
            "vgg16_w_stn_at_drp",
            "vgg16_w_stn_oc_drp",
        ],
        case_sensitive=False,
    ),
    default="vgg16_w_stn_oc_drp",
)
@click.option("--model_state", type=click.Path(exists=True), default=None)
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option(
    "--split_slug",
    type=str,
    required=True,
    help="""This specifies how the data is split into train/
devel/test sets. Balanced/Unbalanced refer to whether selecting
equal number of images from each class. xs, sm, lg, dev all refer
to what fraction is picked for train/devel/test.""",
)
@click.option(
    "--target_metrics",
    type=str,
    default="bt_g",
    help="""Enter the target metrics separated by commas""",
)
@click.option(
    "--loss",
    type=click.Choice(
        [
            "mse",
            "aleatoric",
            "aleatoric_cov",
        ],
        case_sensitive=False,
    ),
    default="aleatoric_cov",
    help="""The loss function to use""",
)
@click.option(
    "--expand_data",
    type=int,
    default=1,
    help="""This controls the factor by which the training
data is augmented""",
)
@click.option("--cutout_size", type=int, default=167)
@click.option("--channels", type=int, default=3)
@click.option(
    "--n_workers",
    type=int,
    default=4,
    help="""The number of workers to be used during the
data loading process.""",
)
@click.option("--batch_size", type=int, default=16)
@click.option("--epochs", type=int, default=40)
@click.option("--lr", type=float, default=5e-7)
@click.option("--momentum", type=float, default=0.9)
@click.option("--weight_decay", type=float, default=0)
@click.option(
    "--parallel/--no-parallel",
    default=True,
    help="""The parallel argument controls whether or not
to use multiple GPUs when they are available""",
)
@click.option(
    "--normalize/--no-normalize",
    default=True,
    help="""The normalize argument controls whether or not, the
loaded images will be normalized using the arsinh function""",
)
@click.option(
    "--label_scaling",
    type=str,
    default="std",
    help="""The label scaling option controls whether to
standardize the labels or not. Set this to std for sklearn's
StandardScaling() and minmax for sklearn's MinMaxScaler().
This is especially important when predicting multiple
outputs""",
)
@click.option(
    "--transform/--no-transform",
    default=True,
    help="""If True, the training images are passed through a
series of random transformations""",
)
@click.option(
    "--crop/--no-crop",
    default=True,
    help="""If True, all images are passed through a cropping
operation before being fed into the network. Images are cropped
to the cutout_size parameter""",
)
@click.option(
    "--repeat_dims/--no-repeat_dims",
    default=True,
    help="""In case of multi-channel data, whether to repeat a two
dimensional image as many times as the number of channels""",
)
@click.option(
    "--nesterov/--no-nesterov",
    default=False,
    help="""Whether to use Nesterov momentum or not""",
)
@click.option(
    "--dropout_rate",
    type=float,
    default=None,
    help="""The dropout rate to use for all the layers in the
    model. If this is set to None, then the default dropout rate
    in the specific model is used.""",
)
def train(**kwargs):
    """Runs the training procedure using MLFlow."""

    # Copy and log args
    args = {k: v for k, v in kwargs.items()}

    # Discover devices
    args["device"] = discover_devices()

    # Create target metrics array
    target_metric_arr = args["target_metrics"].split(",")

    # Calculating the number of outputs
    n_out = len(target_metric_arr)
    if args["loss"] == "aleatoric":
        n_out = int(n_out * 2)
    elif args["loss"] == "aleatoric_cov":
        n_out = int((3 * n_out + n_out**2) / 2)

    # Create the model given model_type
    cls = model_factory(args["model_type"])
    model_args = {
        "cutout_size": args["cutout_size"],
        "channels": args["channels"],
        "n_out": n_out,
    }

    if "drp" in args["model_type"].split("_"):
        logging.info(
            "Using dropout rate of {} in the model".format(
                args["dropout_rate"]
            )
        )
        model_args["dropout"] = "True"

    model = cls(**model_args)
    model = nn.DataParallel(model) if args["parallel"] else model
    model = model.to(args["device"])

    # Chnaging the default dropout rate if specified
    if args["dropout_rate"] is not None:
        specify_dropout_rate(model, args["dropout_rate"])

    # Load the model from a saved state if provided
    if args["model_state"]:
        model.load_state_dict(torch.load(args["model_state"]))

    # Define the optimizer
    optimizer = opt.SGD(
        model.parameters(),
        lr=args["lr"],
        momentum=args["momentum"],
        nesterov=args["nesterov"],
        weight_decay=args["weight_decay"],
    )

    # Define the criterion
    loss_dict = {
        "mse": nn.MSELoss(),
        "aleatoric": AleatoricLoss(average=True),
        "aleatoric_cov": AleatoricCovLoss(
            num_var=len(target_metric_arr), average=True
        ),
    }
    criterion = loss_dict[args["loss"]]

    # Create a DataLoader factory based on command-line args
    loader_factory = partial(
        get_data_loader,
        batch_size=args["batch_size"],
        n_workers=args["n_workers"],
    )

    # Select the desired transforms
    T = None
    T_crop = None

    if args["crop"]:
        T = nn.Sequential(
            K.CenterCrop(args["cutout_size"]),
        )
        T_crop = nn.Sequential(
            K.CenterCrop(args["cutout_size"]),
        )

    if args["transform"]:
        T = nn.Sequential(
            K.CenterCrop(args["cutout_size"]),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            K.RandomRotation(360),
        )

        T_crop = nn.Sequential(
            K.CenterCrop(args["cutout_size"]),
        )

    # Generate the DataLoaders and log the train/devel/test split sizes
    splits = ("train", "devel", "test")
    datasets = {
        k: FITSDataset(
            data_dir=args["data_dir"],
            slug=args["split_slug"],
            cutout_size=args["cutout_size"],
            channels=args["channels"],
            normalize=args["normalize"],
            repeat_dims=args["repeat_dims"],
            label_col=target_metric_arr,
            transform=T if k == "train" else T_crop,
            expand_factor=args["expand_data"] if k == "train" else 1,
            label_scaling=args["label_scaling"],
            split=k,
        )
        for k in splits
    }
    loaders = {k: loader_factory(v) for k, v in datasets.items()}
    args["splits"] = {k: len(v.dataset) for k, v in loaders.items()}

    # Start the training process
    mlflow.set_experiment(args["experiment_name"])
    with mlflow.start_run(run_id=args["run_id"], run_name=args["run_name"]):

        # Write the parameters and model stats to MLFlow
        args = {**args, **model_stats(model)}  # py3.9: d1 |= d2
        for k, v in args.items():
            mlflow.log_param(k, v)

        # Set up trainer
        trainer = create_trainer(
            model, optimizer, criterion, loaders, args["device"]
        )

        # Run trainer and save model state
        trainer.run(loaders["train"], max_epochs=args["epochs"])
        slug = (
            f"{args['experiment_name']}-{args['split_slug']}-"
            f"{mlflow.active_run().info.run_id}"
        )
        model_path = save_trained_model(model, slug)
        logging.info("Saved model to {}".format(model_path))

        # Log model as an artifact
        mlflow.log_artifact(model_path)

        # Visualize spatial transformation
        if args["model_type"] != "vgg16":
            if hasattr(model, "spatial_transform") or hasattr(
                model.module, "spatial_transform"
            ):
                output_dir = Path("output") / slug
                output_dir.mkdir(parents=True, exist_ok=True)
                nrow = round(math.sqrt(args["batch_size"]))
                visualize_spatial_transform(
                    model,
                    loaders["devel"],
                    output_dir,
                    device=args["device"],
                    nrow=nrow,
                )

                # Log output directory as an artifact
                mlflow.log_artifacts(output_dir)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train()
