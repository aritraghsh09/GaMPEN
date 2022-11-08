# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import math
from itertools import chain, zip_longest

import pandas as pd


split_types = dict(
    xs=dict(train=0.027, devel=0.003, test=0.970),
    sm=dict(train=0.045, devel=0.005, test=0.950),
    md=dict(train=0.090, devel=0.010, test=0.900),
    lg=dict(train=0.200, devel=0.050, test=0.750),
    xl=dict(train=0.450, devel=0.050, test=0.500),
    dev=dict(train=0.700, devel=0.150, test=0.150),
    dev2=dict(train=0.700, devel=0.050, test=0.250),
)


def interleave(L):
    return [x for x in chain(*zip_longest(*L)) if x is not None]


def make_splits(x, weights, split_col=None):
    if split_col is not None:  # balanced splits
        splits_list = list(x[split_col].unique())
        by_split = {s: list(x[x[split_col] == s].index) for s in splits_list}
        x = x.iloc[interleave(by_split.values())]
    splits = dict()
    prev_index = 0
    for k, v in weights.items():
        next_index = prev_index + math.ceil((len(x) * v))
        splits[k] = x[prev_index:next_index]
        prev_index = next_index
    return splits


@click.command()
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option("--target_metric", type=str, default="bt_g")
def main(data_dir, target_metric):
    """Generate train/devel/test splits from the dataset provided."""

    # Make the splits directory
    data_dir = Path(data_dir)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Read and shuffle the catalog for the full dataset
    df = pd.read_csv(data_dir / "info.csv")
    df = df.sample(frac=1, random_state=0)

    for balance in [False, True]:
        # Balance if needed
        col = None
        if balance:
            col = "balance"
            df["balance"] = pd.cut(df[target_metric], 4)

        # Generate splits and write to disk
        for split_type in split_types.keys():
            splits = make_splits(df, split_types[split_type], split_col=col)
            split_slug = f"{'un' if not balance else ''}balanced-{split_type}"
            for k, v in splits.items():
                v.to_csv(splits_dir / f"{split_slug}-{k}.csv", index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
