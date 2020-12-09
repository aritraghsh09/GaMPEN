# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm

from ggt.data import FITSDataset, get_data_loader
from ggt.models import model_factory
from ggt.utils import discover_devices

def predict(
    model_path, 
    dataset, 
    cutout_size, 
    channels, 
    parallel=False,
    batch_size=256, 
    n_workers = 1, 
    model_type='ggt',
):
    """Using the model defined in model path, return the output values for the given
    set of images"""

    # Discover devices
    device = discover_devices()

    # Load the model
    logging.info("Loading model...")
    cls = model_factory(model_type)
    model = cls(cutout_size, channels)
    model = nn.DataParallel(model) if parallel else model
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    # Create a data loader
    loader = get_data_loader(
        dataset, batch_size=batch_size, n_workers=n_workers, shuffle=False
    )

    logging.info("Performing predictions...")
    yh = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(loader):
            X, _ = data
            yh.append(model(X.to(device)))
    return torch.cat(yh).cpu().numpy()


@click.command()
@click.option(
    '--model_type',
    type=click.Choice(['ggt','vgg16'], 
    case_sensitive=False),
    default='ggt',
)
@click.option('--model_path', type=click.Path(exists=True), required=True)
@click.option('--output_path', type=click.Path(writable=True), required=True)
@click.option('--data_dir', type=click.Path(exists=True), required=True)
@click.option('--cutout_size', type=int, default=167)
@click.option('--channels', type=int, default=1)
@click.option(
    '--slug', 
    type=str, 
    required=True,
    help='''This specifies which slug (balanced/unbalanced
              xs, sm, lg, dev) is used to perform predictions on.''',
)
@click.option('--split', type=str, required=True, default='test')
@click.option(
    '--normalize/--no-normalize', 
    default=True,
    help='''The normalize argument controls whether or not, the
              loaded images will be normalized using the arcsinh function''',
)
@click.option('--batch_size', type=int, default=256)
@click.option(
    '--n_workers', 
    type=int, 
    default=16,
    help='''The number of workers to be used during the
              data loading process.''',
)
@click.option(
    '--parallel/--no-parallel', 
    default=False,
    help='''The parallel argument controls whether or not
              to use multiple GPUs when they are available''',
)
@click.option('--label_col', type=str, default='bt_g')
@click.option(
    '--repeat_dims/--no-repeat_dims', 
    default=False,
    help='''In case of multi-channel data, whether to repeat a two 
              dimensional image as many times as the number of channels''',
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
    label_col,
    model_type, 
    repeat_dims,
):
    
    # Load the data and create a dataloader
    logging.info("Loading Images to Device")
    dataset = FITSDataset(
        data_dir, 
        slug=slug, 
        normalize=normalize, 
        split=split,
        cutout_size=cutout_size, 
        channels = channels, 
        label_col = label_col,
        repeat_dims=repeat_dims,
    )

    # Making Predictions
    preds = predict(
        model_path, 
        dataset, 
        cutout_size, 
        channels, 
        parallel=parallel,
        batch_size=batch_size, 
        n_workers = n_workers, 
        model_type=model_type,
    ) 

    # Write a CSV of predictions
    catalog = pd.read_csv(
        Path(data_dir) / "splits/{}-{}.csv".format(slug, split)
    )
    catalog["preds"] = preds
    catalog.to_csv(output_path, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
