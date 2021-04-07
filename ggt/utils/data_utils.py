import pandas as pd
from pathlib import Path


def load_cat(data_dir, slug, split):
    """Loads and returns pandas dataframe"""

    data_dir = Path(data_dir)

    if split:
        catalog = data_dir / f"splits/{slug}-{split}.csv"
    else:
        catalog = data_dir / "info.csv"

    return pd.read_csv(catalog)
