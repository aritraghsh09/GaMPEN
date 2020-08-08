from astropy.io import fits
import numpy as np
import pandas as pd
from pathlib import Path
import random
from tqdm import tqdm
import multiprocessing as mp

import torch
from torch.utils.data import Dataset

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')


class FITSDataset(Dataset):
    """Dataset from FITS files."""

    def __init__(self, data_dir, slug=None, split=None, channels=1,
        cutout_size=167, label_col='bt_g', normalize=True, transform=None):

        # Set data directory
        self.data_dir = Path(data_dir)

        # Set cutouts shape
        self.cutout_shape = (channels, cutout_size, cutout_size)

        # Set requested transforms
        self.transform = transform

        # Read the catalog csv file
        if split:
            catalog = self.data_dir / "splits/{}-{}.csv".format(slug, split)
        else:
            catalog = self.data_dir / "info.csv"

        # Define paths
        self.data_info = pd.read_csv(catalog)
        self.cutouts_path = self.data_dir / "cutouts"
        self.tensors_path = self.data_dir / "tensors"
        self.tensors_path.mkdir(parents=True, exist_ok=True)

        # Retrieve labels & filenames
        self.labels = np.asarray(self.data_info[label_col])
        self.filenames = np.asarray(self.data_info["file_name"])

        # If we haven't already generated PyTorch tensor files, generate them
        for filename in tqdm(self.filenames):
            filepath = self.tensors_path / (filename + ".pt")
            if not filepath.is_file():
                t = self.load_fits_as_tensor(self.cutouts_path / filename)
                torch.save(t, filepath)

        # Preload the tensors
        self.observations = map(self.load_tensor, self.filenames)
        self.observations = list(self.observations)  # force eval


    def __getitem__(self, index):
        if isinstance(index, slice):
            # TODO @amritrau stack X and y separately
            # start, stop, step = index.indices(len(self))
            # return [self[i] for i in range(start, stop, step)]
            raise NotImplementedError("Slice as index")
        elif isinstance(index, int):
            # Load image as tensor
            X = self.observations[index]

            # Get image label
            y = torch.tensor(self.labels[index]).unsqueeze(-1)

            # Normalize if necessary
            if self.normalize:
                X = torch.log(X + (X ** 2 + 1) ** 0.5)  # arsinh

            # Transform X and return X, y
            if self.transform:
                X = self.transform(X)
            return X, y
        elif isinstance(index, tuple):
            raise NotImplementedError("Tuple as index")
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))


    def __len__(self):
        return len(self.labels)


    def load_fits_as_tensor(self, filename):
        # Open FITS file and convert to Torch tensor
        fits_np = fits.getdata(filename, memmap=False)
        fits_th = torch.from_numpy(fits_np.astype(np.float32))

        # Reshape
        return fits_th.view(self.cutout_shape)


    def load_tensor(self, filename):
        return torch.load(self.tensors_path / (filename + ".pt"))
