# -*- coding: utf-8 -*-
import click
import logging
import glob
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from tqdm import tqdm

from ggt.data import FITSDataset
from ggt.models import model_factory
from ggt.utils import arsinh_normalize


@click.command()
@click.option('--model_path', type=click.Path(exists=True), required=True)
@click.option('--image_dir', type=click.Path(exists=True), required=True)
@click.option('--cutout_size', type=int, default=167)
@click.option('--channels', type=int, default=1)
def main(model_path, image_dir, cutout_size, channels):
    """Using the spatial transformer layer of the model defined in `model_path`,
    write cropped versions of each image in `image_dir` back to disk."""

    # Load the model
    logging.info("Loading model...")
    cls = model_factory('ggt')
    model = cls(cutout_size, channels)
    model.load_state_dict(torch.load(model_path))

    # Collect all images, then iterate
    images = glob.glob(str(Path(image_dir) / "*.fits"))
    logging.info(f"Cropping {len(images)} images...")
    for path in tqdm(images):
        # Resize and normalize the image
        X = FITSDataset.load_fits_as_tensor(path)[None, :, :]
        X = F.interpolate(X[None, :, :, :], size=cutout_size).squeeze(0)
        X = arsinh_normalize(X).unsqueeze(0)

        # Transform the image
        with torch.no_grad():
            Xt = model.spatial_transform(X)

        # Save the old and new images to disk
        outfile = lambda x: f"{path.replace('.fits', x)}.png"
        save_image(X, outfile('-orig'))
        save_image(Xt, outfile('-crop'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
