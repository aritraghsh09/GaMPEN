from .dataset import FITSDataset

from torch.utils.data import DataLoader


def get_data_loader(dataset, batch_size, n_workers, shuffle=True):
    return DataLoader(dataset, batch_size, shuffle=shuffle,
                      num_workers=n_workers)


__all__ = ['FITSDataset', 'get_data_loader']
