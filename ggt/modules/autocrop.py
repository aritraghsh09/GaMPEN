# -*- coding: utf-8 -*-
import click
import logging
import glob
from pathlib import Path

import torch
import torch.nn as nn
import kornia.augmentation as K

from tqdm import tqdm

from ggt.data import FITSDataset
from ggt.models import model_factory
from ggt.utils import arsinh_normalize, discover_devices
from astropy.io import fits


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
    default="vgg16_w_stn_oc_drp",
)
@click.option("--model_path", type=click.Path(exists=True), required=True)
@click.option("--cutout_size", type=int, default=167)
@click.option("--channels", type=int, default=1)
@click.option("--n_pred", type=int, default=1)
@click.option("--image_dir", type=click.Path(exists=True), required=True)
@click.option("--out_dir", type=click.Path(exists=True), required=True)
@click.option("--normalize/--no-normalize", default=True)
@click.option("--transform/--no-transform", default=True)
@click.option("--repeat_dims/--no-repeat_dims", default=True)
@click.option("--parallel/--no-parallel", default=True)
@click.option("--cov_errors/--no-cov_errors", default=False)
@click.option("--errors/--no-errors", default=False)
def main(
    model_type,
    model_path,
    cutout_size,
    channels,
    n_pred,
    image_dir,
    out_dir,
    normalize,
    transform,
    repeat_dims,
    parallel,
    cov_errors,
    errors,
):
    """Using the spatial transformer layer of the model defined in `model_path`,
    write cropped versions of each image in `image_dir` back to disk."""

    # Select the target device
    device = discover_devices()

    # Calculating the number of outputs
    if errors:
        n_out = int(n_pred * 2)
    elif cov_errors:
        n_out = int((3 * n_pred + n_pred ** 2) / 2)
    else:
        n_out = n_pred

    # Create the model given model_type
    cls = model_factory(model_type)
    model_args = {
        "cutout_size": cutout_size,
        "channels": channels,
        "n_out": n_out,
    }

    if "drp" in model_type.split("_"):
        model_args["dropout"] = "True"

    model = cls(**model_args)
    model = nn.DataParallel(model) if parallel else model
    model = model.to(device)

    # Load the model from a saved state if provided
    model.load_state_dict(torch.load(model_path))

    # Collect all images, then iterate
    images = glob.glob(str(Path(image_dir) / "*.fits"))
    logging.info(f"Cropping {len(images)} images...")

    T = nn.Sequential(
        K.CenterCrop(cutout_size),
    )

    for path in tqdm(images):
        # Resize and normalize the image
        X = FITSDataset.load_fits_as_tensor(path)

        # Normalize if necessary
        if normalize:
            X = arsinh_normalize(X)  # arsinh

        # Crop the image
        if transform:
            X = T(X)  # cropping images to the right size

        if repeat_dims:
            if not transform:
                X = X.unsqueeze(0)
                X = X.repeat(channels, 1, 1)
            else:
                X = X.repeat(1, channels, 1, 1)

        # Transform the image
        with torch.no_grad():
            if parallel:
                Xt = model.module.spatial_transform(X.to(device))
            else:
                Xt = model.spatial_transform(X.to(device))

        file_name = path.split("/")[-1]
        out_path = Path(out_dir) / file_name

        def save_fits_image(np_array, out_path):
            # Save the cropped image to disk
            hdu = fits.PrimaryHDU(np_array)
            hdul = fits.HDUList([hdu])
            hdul.writeto(out_path, overwrite=True)

        # Save the cropped image to disk
        Xt = Xt[0, 0, :, :]
        save_fits_image(Xt.cpu().numpy(), out_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
