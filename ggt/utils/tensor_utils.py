import torch
import numpy as np


def tensor_to_numpy(x):
    """Convert a torch tensor to NumPy for plotting."""
    return np.clip(x.numpy().transpose((1, 2, 0)), 0, 1)


def arsinh_normalize(X):
    """Normalize a Torch tensor with arsinh."""
    return torch.log(X + (X ** 2 + 1) ** 0.5)


def load_tensor(filename, tensors_path):
    """Load a Torch tensor from disk."""
    return torch.load(tensors_path / (filename + ".pt"))


def chunk_seq(seq, size=2048):
    """Break a sequence into chunks of a given size. 
    From https://stackoverflow.com/questions/434287."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))