# -*- coding: utf-8 -*-
import click
import logging
import glob
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from astropy.io import fits
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from ggt.data import FITSDataset
from ggt.models import model_factory
from ggt.utils import tensor_to_numpy, arsinh_normalize


@click.command()
@click.option('--model_path', type=click.Path(exists=True), required=True)
@click.option('--image_dir', type=click.Path(exists=True), required=True)
def main(model_path, image_dir, model_image_dim=167):
    """Using the spatial transformer layer of the model defined in `model_path`,
    write cropped versions of each image in `image_dir` back to disk."""

    # Load the model
    logging.info("Loading model...")
    cls = model_factory('ggt')
    model = cls()
    model.load_state_dict(torch.load(model_path))

    # Collect all images, then iterate
    images = glob.glob(str(Path(image_dir) / "*.fits"))
    logging.info(f"Cropping {len(images)} images...")
    for path in tqdm(images):
        # Resize and normalize the image
        X = FITSDataset.load_fits_as_tensor(path)[None, :, :]
        X = F.interpolate(X[None, :, :, :], size=model_image_dim).squeeze(0)
        X = arsinh_normalize(X).unsqueeze(0)

        # Transform the image
        with torch.no_grad():
            Xt = model.spatial_transform(X)

        # Save the image to disk
        outfile = f"{path.replace('.fits', '')}.png"
        save_image(Xt, outfile)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
