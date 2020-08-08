# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import math
from itertools import chain, zip_longest

import pandas as pd
from tqdm import tqdm


split_types = dict(
    xs=dict(train=0.027, devel=0.003, test=0.970),
    sm=dict(train=0.045, devel=0.005, test=0.950),
    md=dict(train=0.090, devel=0.010, test=0.900),
    lg=dict(train=0.200, devel=0.050, test=0.750),
    xl=dict(train=0.450, devel=0.050, test=0.500)
)


def interleave(L):
    return [x for x in chain(*zip_longest(*L)) if x is not None]


def make_splits(x, ws, class_col=None):
    if class_col is not None:  # balanced splits
        classes = list(x[class_col].unique())
        by_class = {cls: list(x[x[class_col] == cls].index) for cls in classes}
        x = x.iloc[interleave(by_class.values())]
    splits = dict()
    prev_index = 0
    for k, v in ws.items():
        next_index = prev_index + math.ceil((len(x) * v))
        splits[k] = x[prev_index:next_index]
        prev_index = next_index
    return splits


@click.command()
@click.option('--data_dir', type=click.Path(exists=True), required=True)
@click.option('--split_type', type=click.Choice(split_types.keys()))
@click.option('--split_slug', type=str, required=True)
@click.option('--balance/--no-balance', default=False)
def main(data_dir, split_type, split_slug, balance):
    """Generate train/devel/test splits from the dataset provided.
    """
    logger = logging.getLogger(__name__)

    # Make the splits directory
    data_dir = Path(data_dir)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Read and shuffle the catalog for the full dataset
    df = pd.read_csv(data_dir / "info.csv")
    df = df.sample(frac=1, random_state=0)

    # Balance if needed
    class_col = None
    if balance:
        class_col = 'balance'
        df['balance'] = pd.cut(df['bt_g'], 4)

    # Generate splits and write to disk
    splits = make_splits(df, split_types[split_type], class_col=class_col)
    for k, v in splits.items():
        v.to_csv(splits_dir / f"{split_slug}-{k}.csv", index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
