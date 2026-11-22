# Bike Sharing Demand Prediction

Introduction to Machine Learning - course project.

**Team:** Lujain (Project coordinator), Nouf (Data lead), Ajwan (Modelling lead), Ghaleh (Evaluation and reporting lead)

## Problem
Predict `cnt`, the number of bikes rented per hour, from calendar and weather features, to help a bike-sharing operator plan bike redistribution. Task type: **regression**. Main metric: **RMSE** (MAE and R² also reported).

## Data
UCI Bike Sharing, ID 275, hourly file `hour.csv` - 17,379 rows, Washington D.C., 2011-2012.
Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894 - licence CC BY 4.0.

`data/hour.csv` is included in the repository. To download it again:
```bash
python data/download_data.py
```
If the download fails: open https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset, click **Download**, unzip, and copy `hour.csv` into `data/`.

## How to run
**Locally** (Python 3.10+):
```bash
pip install -r requirements.txt
jupyter notebook
```
Open the notebooks in `notebooks/` and run them **in order** (01 -> 04), each with *Run All*.

**Google Colab:**
```python
!git clone <repository-url>
%cd <repository-name>/notebooks
```
Then run the cells of each notebook in order. Notebook 03 (grid search) takes several minutes.

## Folder structure
```
README.md
requirements.txt
contribution_log.csv
data/download_data.py              downloads hour.csv and prints a first check
data/hour.csv                      the data set
src/utils.py                       split, preprocessing pipelines, CV and metrics (shared)
notebooks/
  01_eda_preprocessing.ipynb       data description, quality checks, 6 figures, pipeline
  02_models.ipynb                  baseline, linear (raw/log target), Ridge, k-NN, decision tree
  03_ensembles_evaluation.ipynb    random forest, gradient boosting, tuning, test, error analysis
  04_extension.ipynb               neural network (MLP)
report/
  Bike_Sharing_Project_Proposal.pdf
  Bike_Sharing_Final_Report.docx / .pdf
  figures/                         created by the notebooks
  results/                         CSV/JSON results created by the notebooks
presentation/
  Bike_Sharing_Presentation.pptx
```

## Method
- **Chronological split:** rows sorted by date + hour; earliest 80% = training, latest 20% = test. EDA uses training data only.
- **No leakage:** `casual` and `registered` removed (they sum to `cnt`); all preprocessing inside `Pipeline` + `ColumnTransformer`.
- **Preprocessing:** linear models/k-NN/MLP use one-hot codes + scaled numbers; tree models use raw integer codes.
- **Validation:** `TimeSeriesSplit` (5 folds, 24-row gap) on training data only; mean and std reported.
- **Models:** mean baseline, linear regression (raw and log target), Ridge, k-NN, decision tree, random forest, gradient boosting; best ensemble tuned with `GridSearchCV`.
- **Test set:** used once, at the end of notebook 03.
- **Extension:** MLP neural network, compared with CV only.
- `random_state = 42` everywhere.

## Main results
Five-fold time-series CV on the training period (validation RMSE, bikes per hour):

| Model | CV RMSE (mean ± std) |
|---|---|
| Mean baseline | 180.6 ± 42.7 |
| Linear Regression | 109.0 ± 19.5 |
| Ridge | 109.1 ± 19.6 |
| Linear Regression (log target) | 113.5 ± 30.8 |
| k-NN (k=10) | 132.8 ± 28.5 |
| Decision Tree | 95.4 ± 21.5 |
| Gradient Boosting | 89.6 ± 18.4 |
| **Random Forest** | **79.4 ± 19.3** |
| Neural network (extension, best size) | 111.1 ± 32.8 |

Final model: tuned **Random Forest** (200 trees, unlimited depth), evaluated once on the test period (7 Aug - 31 Dec 2012):

| | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest - test | 45.2 | 69.9 | 0.90 |
| Mean baseline - test | 175.0 | 232.6 | -0.11 |

Largest errors: working-day rush hours, rainy hours and special days (Christmas Eve, day after Thanksgiving, Hurricane Sandy). Full discussion in `report/Bike_Sharing_Final_Report.pdf`.

## How to reproduce
Run the four notebooks in order with *Run All*. All random seeds are fixed (`random_state = 42`), so the numbers above are reproduced; small differences can appear with other library versions. Tested with pandas 3.0, scikit-learn 1.9, numpy 2.5, matplotlib 3.11, seaborn 0.13.

## Limitations
Observed (not forecast) weather; one city and two years only; unusual events not represented. See section 7 of the report.
