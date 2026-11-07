Bike Sharing Demand Prediction
Machine Learning Course Project — Team of 4
This project predicts the hourly number of rented bikes (`cnt`) using time, calendar, and weather information from the UCI Bike Sharing dataset (ID 275).
Problem
Bike-sharing operators need short-term demand estimates to plan bicycle redistribution, staffing, maintenance, and station capacity. Given information available before an hour begins—such as hour, day type, season, weather condition, temperature, humidity, and wind speed—the model predicts the number of bikes rented during that hour.
Task type: Regression
Target variable: `cnt`
Main metrics: MAE, RMSE, and R²
Primary metric: RMSE, because large demand-prediction errors are more costly for operational planning.
Dataset
Name: Bike Sharing — hourly data (`hour.csv`)
Source: UCI Machine Learning Repository, Dataset ID 275
Link: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
Size: 17,379 hourly records and 16 columns
Target: `cnt` — total bikes rented in one hour
Important leakage prevention: `casual` and `registered` must not be used as input features because their sum equals the target `cnt`.
Repository structure
```text
bike-sharing-project/
├── README.md
├── requirements.txt
├── contribution_log.md
├── data/
│   └── download_data.py
├── notebooks/
│   ├── 01_eda_preprocessing.py
│   ├── 02_models.py
│   ├── 03_ensembles_evaluation.py
│   └── 04_extension.py
├── src/
│   └── utils.py
└── report/
    └── final_report.md
```
How to run
Install dependencies:
```bash
pip install -r requirements.txt
```
Download the dataset:
```bash
python data/download_data.py
```
Or manually download `hour.csv` from the UCI page and place it in:
```text
data/hour.csv
```
Run the notebooks/scripts in order:
```text
01_eda_preprocessing.py
02_models.py
03_ensembles_evaluation.py
04_extension.py
```
All experiments use `random_state=42` for reproducibility.
Models
Model	Purpose
Mean predictor	Baseline reference
Linear Regression	Simple interpretable linear model
Decision Tree Regressor	Captures non-linear relationships
K-Nearest Neighbors Regressor	Local pattern-based model
Random Forest Regressor	Bagging ensemble
Gradient Boosting Regressor	Boosting ensemble
The best ensemble model is tuned using `GridSearchCV` on the training set only. The untouched test set is used once at the end.
Limitations
The dataset covers a specific city and period, so results may not transfer directly to another city.
Sudden events, holidays not represented well in the data, infrastructure changes, and extreme weather can reduce accuracy.
The model should support planning decisions; it should not be used as the only basis for safety-critical operational decisions.
Team roles
Member	Role
Lujain	Project coordinator
Nouf	Data and EDA lead
Ajwan	Modeling lead
Ghaleh	Evaluation and reporting lead
Every member contributes to code, report, and presentation.
References
UCI Machine Learning Repository. Bike Sharing Dataset. Dataset ID 275. https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
Fanaee-T, H., & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. Progress in Artificial Intelligence, 2(2–3), 113–127.
scikit-learn documentation: https://scikit-learn.org/stable/
pandas documentation: https://pandas.pydata.org/docs/
