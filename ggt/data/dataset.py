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
        self.normalize = normalize
        self.transform = transform

        # Read the catalog csv file
        if split:
            catalog = self.data_dir / f"splits/{slug}-{split}.csv"
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
        self.observations = [self.load_tensor(f) for f in tqdm(self.filenames)]


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
            y = torch.tensor(self.labels[index]).unsqueeze(-1).float()

            # Normalize if necessary
            if self.normalize:
                X = torch.log(X + (X ** 2 + 1) ** 0.5)  # arsinh

            # Transform and reshape X
            if self.transform:
                X = self.transform(X)
            X = X.view(self.cutout_shape).float()
            if X.type() != 'torch.FloatTensor':
                print(X.type())
                raise ValueError

            # Return X, y
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
        return torch.from_numpy(fits_np.astype(np.float32))


    def load_tensor(self, filename):
        return torch.load(self.tensors_path / (filename + ".pt"))
