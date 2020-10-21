# -*- coding: utf-8 -*-
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

    #Creating a Data Loader
    loader = get_data_loader(dataset, batch_size=batch_size, n_workers=n_workers,\
        shuffle=False)

    output_values = []

    for data in tqdm(loader):
        img = data[0]
        with torch.no_grad():
            output_values.append(model(img.to(device)))

    return np.array(torch.cat(output_values).cpu())


def main(model_path, cat_out_path, data_dir, cutout_size, channels,\
    parallel=False, slug='balanced-dev', split='test', normalize=True,\
    batch_size=256, n_workers = 1, label_col='bt_g'):
    
    # Load the data and create a dataloader
    logging.info("Loading Images to Device")
    dataset = FITSDataset(data_dir, slug=slug, normalize=normalize, split=split,\
        cutout_size=cutout_size, channels = channels, label_col = label_col)

    # Making Predictions
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