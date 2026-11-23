"""Shared helpers for the Bike Sharing project.

Every notebook imports from here so that the data split, the preprocessing
and the metrics are defined once and are identical everywhere.
"""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- constants
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "hour.csv"
FIGURES = ROOT / "report" / "figures"
RESULTS = ROOT / "report" / "results"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42                 # fixed everywhere for reproducibility
TEST_FRACTION = 0.20      # latest 20% of the timeline is the held-out test set
DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

TARGET = "cnt"
# casual + registered = cnt, so using them would leak the answer.
# instant is a row id and dteday is the raw date (already split into yr/mnth/weekday).
DROP_COLUMNS = ["cnt", "casual", "registered", "instant", "dteday", "timestamp"]

NUMERIC = ["temp", "atemp", "hum", "windspeed", "yr"]
CATEGORICAL = ["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]


# ---------------------------------------------------------------- data
def download_data():
    """Download the UCI zip and extract hour.csv into data/."""
    DATA_FILE.parent.mkdir(exist_ok=True)
    archive = DATA_FILE.parent / "bike_sharing.zip"
    print("Downloading", DATA_URL)
    urlretrieve(DATA_URL, archive)
    with ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith("hour.csv"))
        DATA_FILE.write_bytes(z.read(name))
    print("Saved", DATA_FILE)


def add_timestamp(df):
    """Convert the date column and build a full timestamp (date + hour).
    The timestamp gives the true order of the observations for the split."""
    df = df.copy()
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["timestamp"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    return df


def check_missing(df):
    """Missing values and duplicates in one small table."""
    report = pd.DataFrame({"missing": df.isna().sum(),
                           "missing_%": (df.isna().mean() * 100).round(2)})
    print(f"Rows: {len(df):,} | duplicate rows: {df.duplicated().sum()} | "
          f"total missing values: {int(report['missing'].sum())}")
    return report[report["missing"] > 0]


def load_data():
    """Read hour.csv (downloading it if needed) and add a full timestamp."""
    if not DATA_FILE.exists():
        download_data()
    df = pd.read_csv(DATA_FILE)
    if len(df) != 17379:
        print(f"WARNING: expected 17,379 rows, found {len(df):,}. Check the file.")
    return add_timestamp(df).sort_values("timestamp").reset_index(drop=True)


def load_split():
    """Chronological split: earliest 80% = train, latest 20% = test.

    A random split would let the model 'see the future' (hours from the same
    days end up in both sets), which makes the test score too optimistic.
    """
    df = load_data()
    cut = int(len(df) * (1 - TEST_FRACTION))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def xy(df):
    """Split a frame into model inputs X and target y."""
    return df.drop(columns=DROP_COLUMNS), df[TARGET]


# ---------------------------------------------------------------- preprocessing
def preprocessing(kind="linear"):
    """Return an unfitted ColumnTransformer.

    kind="linear": one-hot encode the codes and scale numbers. Needed by linear
                   models, k-NN and neural networks (distance / gradient based).
    kind="tree":   keep the integer codes as they are. Trees split on thresholds,
                   so scaling is unnecessary and the hour order (0..23) is useful.
    The transformer is placed inside a Pipeline, so it is fitted on the
    training folds only and never sees validation or test rows.
    """
    if kind == "tree":
        return ColumnTransformer([
            ("all", SimpleImputer(strategy="most_frequent"), NUMERIC + CATEGORICAL),
        ])
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore",
                                                     sparse_output=False))])
    return ColumnTransformer([("numeric", numeric, NUMERIC),
                              ("categorical", categorical, CATEGORICAL)])


def pipeline(model, kind="linear"):
    return Pipeline([("preprocess", preprocessing(kind)), ("model", model)])


# ---------------------------------------------------------------- evaluation
def time_cv():
    """5 expanding-window folds; a 24-row gap reduces leakage between
    neighbouring hours at the edge of each fold."""
    return TimeSeriesSplit(n_splits=5, gap=24)


SCORING = {"MAE": "neg_mean_absolute_error",
           "RMSE": "neg_root_mean_squared_error",
           "R2": "r2"}


def cv_scores(name, estimator, X, y):
    """Cross-validate on the training data only; return mean and std."""
    s = cross_validate(estimator, X, y, cv=time_cv(), scoring=SCORING,
                       return_train_score=True, n_jobs=-1)
    row = {"Model": name}
    for metric in SCORING:
        sign = 1 if metric == "R2" else -1          # sklearn returns negative errors
        for part, label in [("train", "train"), ("test", "val")]:
            values = sign * s[f"{part}_{metric}"]
            row[f"{label}_{metric}_mean"] = values.mean()
            row[f"{label}_{metric}_std"] = values.std()
    row["fit_time_s"] = s["fit_time"].mean()
    return row


def metrics(actual, predicted):
    return {"MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted))}
