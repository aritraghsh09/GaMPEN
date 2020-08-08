# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from ggt.data import FITSDataset, get_data_loader
from ggt.models import model_factory
from ggt.utils import discover_devices

@click.command()
@click.option('--model_path', type=click.Path(exists=True), required=True)
@click.option('--data_dir', type=click.Path(exists=True), required=True)
@click.option('--split_slug', type=str, required=True)
@click.option('--split',
    type=click.Choice(['train', 'devel', 'test']), default='devel')
@click.option('--batch_size', type=int, default=36)
@click.option('--nrow', type=int, default=6)
@click.option('--n_workers', type=int, default=8)
@click.option('--normalize/--no-normalize', default=True)
def main(model_path, data_dir, split_slug, split, batch_size, nrow, n_workers,
    normalize):
    """Visualize the transformation performed by the spatial transformer
    module.
    """
    logger = logging.getLogger(__name__)

    # Select the target device
    device = discover_devices()

    # Create the model given model_type
    cls = model_factory('ggt')  # TODO @amritrau This can be cleaner
    model = cls()
    model = model.to(device)

    # Load the model from a saved state if provided
    model.load_state_dict(torch.load(model_path))

    # Build a DataLoader to pull a batch from the desired split
    dataset = FITSDataset(data_dir=data_dir, slug=split_slug,
        normalize=normalize, split=split)
    loader = get_data_loader(dataset,
        batch_size=batch_size, n_workers=n_workers)

    # Turn off gradients
    with torch.no_grad():
        # Retrieve a batch from the dataloader
        data = next(iter(loader))[0].to(device)

        # Bring the batch onto the CPU
        in_tensor = data.cpu()

        # Execute the predicted spatial transformation
        out_tensor = model.spatial_transform(data).cpu()

        # Helper function to convert a torch tensor to NumPy for plotting
        tensor2np = lambda x: np.clip(x.numpy().transpose((1, 2, 0)), 0, 1)

        # Make grids
        in_grid = tensor2np(torchvision.utils.make_grid(
            in_tensor[:nrow*nrow,:,:,:], nrow=nrow, pad_value=1))
        out_grid = tensor2np(torchvision.utils.make_grid(
            out_tensor[:nrow*nrow,:,:,:], nrow=nrow, pad_value=1))

        # Plot the results side-by-side
        plt.figure(figsize=(15,15), dpi=250)
        plt.imshow(in_grid)
        plt.savefig(f"in_grid-{args.model_name}.png")

        plt.figure(figsize=(15,15), dpi=250)
        plt.imshow(out_grid)
        plt.savefig(f"out_grid-{args.model_name}.png")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
