import torch
import numpy as np
import pandas as pd
from pathlib import Path

# !! ADD sklearn to install_requires !!
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def tensor_to_numpy(x):
    """Convert a torch tensor to NumPy for plotting."""
    return np.clip(x.numpy().transpose((1, 2, 0)), 0, 1)


def arsinh_normalize(X):
    """Normalize a Torch tensor with arsinh."""
    return torch.log(X + (X ** 2 + 1) ** 0.5)


def load_tensor(filename, tensors_path, as_numpy=True):
    """Load a Torch tensor from disk."""
    return torch.load(tensors_path / (filename + ".pt")).numpy()


def standardize_labels(labels, data_dir, split, slug, label_col, scaling):
    """ Standardizes the labels"""

    data_dir = Path(data_dir)

    if split:
        fit_catalog = data_dir / f"splits/{slug}-train.csv"
    else:
        fit_catalog = data_dir / "info.csv"

    fit_data = pd.read_csv(fit_catalog)
    fit_labels = np.asarray(fit_data[label_col])

    if scaling == "std":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Scaling {} is not available.".format(scaling))

    scaler.fit(fit_labels)

    return scaler.transform(labels)
