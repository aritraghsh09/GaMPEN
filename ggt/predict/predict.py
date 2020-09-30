
# -*- coding: utf-8 -*-
import logging
import glob
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from tqdm import tqdm

from ggt.data import FITSDataset
from ggt.models import model_factory
from ggt.utils import arsinh_normalize

## THE FUNCTION BELOW NEEDS TO BE MODIFIED USING A DATA LOADER
## AND CUDA TO SPEED UP THE PROCESS OF MAKING PREDICTIONS.
## See Imagenet example and Amirt's train.py file

def predict(model_path, image_dir, cutout_size, channels):
    """Using the model defined in model path, return the output values for the given
    set of images"""

    # Load the model
    logging.info("Loading model...")
    cls = model_factory('ggt')
    model = cls(cutout_size, channels)
    model.load_state_dict(torch.load(model_path))

    # Collect all images, then iterate
    images = glob.glob(str(Path(image_dir) / "*.fits"))
    logging.info(f"Performing prediction on {len(images)} images...")
    
    output_values = []


    for path in tqdm(images):
        # Resize and normalize the image
        X = FITSDataset.load_fits_as_tensor(path)[None, :, :]
        X = F.interpolate(X[None, :, :, :], size=cutout_size).squeeze(0)
        X = arsinh_normalize(X).unsqueeze(0)

        # Transform the image
        with torch.no_grad():
            output_values.append(model(X))

    return np.array(output_values)

