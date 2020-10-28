# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from tqdm import tqdm

from ggt.data import FITSDataset, get_data_loader
from ggt.models import model_factory
from ggt.utils import arsinh_normalize
from ggt.utils import discover_devices

def predict(model_path, dataset, cutout_size, channels, parallel=False, \
    batch_size=256, n_workers = 1):
    """Using the model defined in model path, return the output values for the given
    set of images"""

    # Discover devices 
    device = discover_devices()

    # Load the model
    logging.info("Loading model...")
    cls = model_factory('ggt')
    model = cls(cutout_size, channels)
    model = nn.DataParallel(model) if parallel else model
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    # Create a data loader
    loader = get_data_loader(dataset, batch_size=batch_size, n_workers=n_workers,\
        shuffle=False)

    output_values = []

    logging.info("Performing predictions...")
    for data in tqdm(loader):
        img = data[0]
        with torch.no_grad():
            output_values.append(model(img.to(device)))

    return np.array(torch.cat(output_values).cpu())

@click.command()
@click.option('--model_path', type=click.Path(exists=True), required=True)
@click.option('--output_path', type=click.Path(writable=True), required=True)
@click.option('--data_dir', type=click.Path(exists=True), required=True)
@click.option('--cutout_size', type=int, default=167)
@click.option('--channels', type=int, default=1)
@click.option('--slug', type=str, required=True,
help='''This specifies which slug (balanced/unbalanced 
xs, sm, lg, dev) is used to perform predictions on.''')
@click.option('--split', type=str, required=True, default='test')
@click.option('--normalize/--no-normalize', default=True,
help='''The normalize argument controls whether or not, the
loaded images will be normalized using the arcsinh function''')
@click.option('--batch_size', type=int, default=256)
@click.option('--n_workers', type=int, default=16,
help='''The number of workers to be used during the
data loading process.''')
@click.option('--parallel/--no-parallel', default=False,
help='''The parallel argument controls whether or not 
to use multiple GPUs when they are available''')
@click.option('--label_col', type=str, default='bt_g')
def main(model_path, cat_out_path, data_dir, cutout_size, channels,\
    parallel, slug, split, normalize, batch_size, n_workers, label_col):
    
    # Load the data and create a data loader
    logging.info("Loading Images to Device")
    dataset = FITSDataset(data_dir, slug=slug, normalize=normalize, split=split,\
        cutout_size=cutout_size, channels = channels, label_col = label_col)

    # Make predictions
    preds = predict(model_path, dataset, cutout_size, channels, parallel=parallel,\
        batch_size=batch_size, n_workers = n_workers) 

    # Outputting a CSV Prediction Catalogue
    catalogue = pd.read_csv(Path(data_dir) / "splits/{}-{}.csv".format(slug, split))
    catalogue["preds"] = preds
    catalogue.to_csv(cat_out_path,index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
