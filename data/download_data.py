"""Download UCI Bike Sharing (ID 275), save data/hour.csv and print a first check.

Run from the project root:  python data/download_data.py
If the download fails, open https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset,
press Download, unzip, and copy hour.csv into this data/ folder, then run this script
again to check the file.
"""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
FOLDER = Path(__file__).resolve().parent
DATA_FILE = FOLDER / "hour.csv"


def download():
    archive = FOLDER / "bike_sharing.zip"
    print("Downloading", DATA_URL)
    urlretrieve(DATA_URL, archive)
    with ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith("hour.csv"))
        DATA_FILE.write_bytes(z.read(name))
    archive.unlink()                      # keep only hour.csv
    print("Saved", DATA_FILE)


if __name__ == "__main__":
    if DATA_FILE.exists():
        print("Found local file:", DATA_FILE)
    else:
        download()

    df = pd.read_csv(DATA_FILE)
    print("\nShape (rows, columns):", df.shape, "- expected 17,379 rows")
    print("\nColumns and types:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("\nTotal missing values:", int(df.isna().sum().sum()))
