# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import torch
import torchvision

from ggt.data import FITSDataset, get_data_loader
from ggt.models import model_factory
from ggt.utils import discover_devices, tensor_to_numpy


def visualize_spatial_transform(
    model,
    loader,
    output_dir,
    device="cpu",
    nrow=6,
    return_grids=False,
    matplotlib_backend="agg",
):

    # Turn off gradients
    with torch.no_grad():
        # Retrieve a batch from the dataloader
        data = next(iter(loader))[0].to(device)

        # Bring the batch onto the CPU
        in_tensor = data.cpu()

        # Execute the predicted spatial transformation
        if hasattr(model, "spatial_transform"):
            out_tensor = model.spatial_transform(data).cpu()
        elif hasattr(model.module, "spatial_transform"):
            out_tensor = model.module.spatial_transform(data).cpu()
        else:
            raise ValueError("Model does not have a spatial_transform method")

        # Make grids
        in_grid = tensor_to_numpy(
            torchvision.utils.make_grid(
                in_tensor[: nrow * nrow, :, :, :], nrow=nrow, pad_value=1
            )
        )
        out_grid = tensor_to_numpy(
            torchvision.utils.make_grid(
                out_tensor[: nrow * nrow, :, :, :], nrow=nrow, pad_value=1
            )
        )

        # If requested, return the grids
        if return_grids:
            return in_grid, out_grid

        if matplotlib_backend is not None:
            matplotlib.use(matplotlib_backend)

        # Show the results and save them to disk
        plt.figure(figsize=(15, 15), dpi=250)
        plt.imshow(in_grid)
        plt.savefig(output_dir / "stn-in_grid.png")

        plt.figure(figsize=(15, 15), dpi=250)
        plt.imshow(out_grid)
        plt.savefig(output_dir / "stn-out_grid.png")

    # Return the output directory containing the images
    return output_dir


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
@click.option("--cutout_size", type=int, default=167)
@click.option("--channels", type=int, default=1)
@click.option("--n_out", type=int, default=1)
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option("--split_slug", type=str, required=True)
@click.option(
    "--split", type=click.Choice(["train", "devel", "test"]), default="devel"
)
@click.option("--batch_size", type=int, default=36)
@click.option("--nrow", type=int, default=6)
@click.option("--n_workers", type=int, default=8)
@click.option("--normalize/--no-normalize", default=True)
@click.option("--dropout", type=float, default=0.0)
def main(
    model_type,
    model_path,
    cutout_size,
    channels,
    n_out,
    data_dir,
    split_slug,
    split,
    batch_size,
    nrow,
    n_workers,
    normalize,
    dropout,
):
    """Visualize the transformation performed by the spatial transformer
    module.
    """

    # Select the target device
    device = discover_devices()

    # Create the model given model_type
    cls = model_factory(model_type)
    model_args = {
        "cutout_size": cutout_size,
        "channels": channels,
        "n_out": n_out,
    }

    if model_type == "vgg16_w_stn_drp":
        model_args["dropout"] = "True"

    model = cls(**model_args)
    model = model.to(device)

    # Load the model from a saved state if provided
    model.load_state_dict(torch.load(model_path))

    # Build a DataLoader to pull a batch from the desired split
    dataset = FITSDataset(
        data_dir=data_dir, slug=split_slug, normalize=normalize, split=split
    )
    loader = get_data_loader(
        dataset, batch_size=batch_size, n_workers=n_workers
    )

    # Determine output filepath
    basename = Path(model_path).stem
    output_dir = Path("output") / basename
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the visualizations
    visualize_spatial_transform(model, loader, output_dir, device, nrow)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
