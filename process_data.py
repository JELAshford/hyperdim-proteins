"""Load in all the original data-blobs and process them into feather files"""

from pathlib import Path
import pandas as pd

DATA_DIR = "./data"


def load_data_folder(directory: str):
    """Load full folder of data blobs and convert to singkle pandas DataFrame."""
    train_files = list(Path(directory).glob("data-*"))
    return pd.concat([*map(pd.read_csv, train_files)])


for dset in ["train", "dev", "test"]:
    data = load_data_folder(f"{DATA_DIR}/{dset}")
    data.to_feather(f"data/{dset}.feather")
