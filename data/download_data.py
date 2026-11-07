from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
folder = Path(__file__).resolve().parent
archive = folder / "bike_sharing.zip"
urlretrieve("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip", archive)
with ZipFile(archive) as z:
    for name in z.namelist():
        if name.endswith("hour.csv"):
            (folder / "hour.csv").write_bytes(z.read(name))
print("Saved", folder / "hour.csv")
