from astropy.io import fits
import numpy as np
import pandas as pd
from functools import partial
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from ggt.utils import arsinh_normalize, load_tensor, chunk_seq

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')


class FITSDataset(Dataset):
    """Dataset from FITS files. Pre-caches FITS files as PyTorch tensors to
    improve data load speed."""

    def __init__(self, data_dir, slug=None, split=None, channels=1,
                 cutout_size=167, label_col='bt_g', normalize=True,
                 transform=None, expand_factor=1):

        # Set data directory
        self.data_dir = Path(data_dir)

        # Set cutouts shape
        self.cutout_shape = (channels, cutout_size, cutout_size)

        # Set requested transforms
        self.normalize = normalize
        self.transform = transform

        # Set data expansion factor (must be an int and >= 1)
        self.expand_factor = expand_factor

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
        logging.info(f"Generating PyTorch tensors from FITS files...")
        for filename in tqdm(self.filenames):
            filepath = self.tensors_path / (filename + ".pt")
            if not filepath.is_file():
                load_path = self.cutouts_path / filename
                t = FITSDataset.load_fits_as_tensor(load_path)
                torch.save(t, filepath)

        # Preload the tensors in chunks to avoid torch mp bug
        chunk_size = 10000
        logging.info(f"Preloading tensors in chunks of {chunk_size}...")
        load_fn = partial(load_tensor, tensors_path=self.tensors_path)
        self.observations = []  # append to self.observations
        for chunk in chunk_seq(self.filenames, chunk_size):
            with mp.Pool(mp.cpu_count() // 2) as p:
                self.observations += list(tqdm(p.imap(load_fn, chunk), total=len(chunk)))
        
        self.observations = [torch.from_numpy(x) for x in self.observations]

    def __getitem__(self, index):
        """Magic method to index into the dataset."""
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(index, int):
            # Load image as tensor ("wrap around")
            X = self.observations[index % len(self.observations)]

            # Get image label ("wrap around"; make sure to cast to float!)
            y = torch.tensor(self.labels[index % len(self.labels)])
            y = y.unsqueeze(-1).float()

            # Normalize if necessary
            if self.normalize:
                X = arsinh_normalize(X)  # arsinh

            # Transform and reshape X
            if self.transform:
                X = self.transform(X)
            X = X.view(self.cutout_shape).float()

            # Return X, y
            return X, y
        elif isinstance(index, tuple):
            raise NotImplementedError("Tuple as index")
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

    def __len__(self):
        """Return the effective length of the dataset."""
        return len(self.labels) * self.expand_factor

    @staticmethod
    def load_fits_as_tensor(filename):
        """Open a FITS file and convert it to a Torch tensor."""
        fits_np = fits.getdata(filename, memmap=False)
        return torch.from_numpy(fits_np.astype(np.float32))
