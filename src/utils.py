from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
REPORT.mkdir(exist_ok=True)
SEED = 42

def load_split():
    df = pd.read_csv(ROOT / "data/hour.csv")
    df["timestamp"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * .8)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def xy(df):
    return df.drop(columns=["cnt", "casual", "registered", "instant", "dteday", "timestamp"]), df["cnt"]

def preprocessing():
    numeric = ["temp", "atemp", "hum", "windspeed", "yr"]
    categorical = ["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
         ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)
    ])

def pipeline(model):
    return Pipeline([("preprocess", preprocessing()), ("model", model)])

def metrics(actual, predicted):
    return {"MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)), "R2": r2_score(actual, predicted)}
