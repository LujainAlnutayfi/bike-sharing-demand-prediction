from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import *
import matplotlib.pyplot as plt
import seaborn as sns
train, _ = load_split()
print("Training shape:", train.shape)
print("Missing values:", train.isna().sum())
print("Duplicate rows:", train.duplicated().sum())
train.describe().to_csv(REPORT / "training_summary.csv")
plots = [("hr", "Demand by hour"), ("mnth", "Demand by month"),
         ("workingday", "Demand by working day"), ("weathersit", "Demand by weather")]
for col, title in plots:
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=train, x=col, y="cnt")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(REPORT / f"fig_{col}.png", dpi=160)
    plt.close()
plt.figure(figsize=(8,4))
sns.histplot(train["cnt"], bins=40)
plt.title("Training target distribution")
plt.tight_layout()
plt.savefig(REPORT / "fig_target_distribution.png", dpi=160)
plt.close()
print("Write interpretations from the actual plots; do not assume findings.")
X, y = xy(train)
print("Model features:", list(X.columns))
print("Preprocessor:", preprocessing())
