# Predictive Modeling (ANN)

Artificial Neural Network that predicts customer churn for the telecom dataset (Stage 3-1 deliverables).

## Model
- scikit-learn `MLPClassifier` (feed-forward ANN) in a `Pipeline` with `StandardScaler`
- Architecture: **11 inputs → 32 (tanh) → 16 (tanh) → 1 (sigmoid)**, Adam optimiser, L2 alpha = 0.1, batch size 64
- Picked by 5-fold cross-validation over 72 configurations. Trained epoch by epoch with early stopping (best epoch 115)
- Decision threshold 0.24, tuned on the validation set to maximise F1

## Test-set results (1,409 customers)
| Metric | ANN | Logistic regression |
|---|---|---|
| ROC-AUC | 0.807 | 0.807 |
| Recall | 0.757 | 0.650 |
| Precision | 0.456 | 0.528 |
| F1 | 0.569 | 0.583 |

Top drivers: **tenure** (by far), then **MonthlyCharges**; SeniorCitizen has a small effect. Contract type has almost no effect in this dataset.

## Contents
- `03_Predictive_Modeling.ipynb` / `.html`: full analysis report
- `Model/`: `ann_churn_pipeline.joblib` (scaler + ANN; takes raw encoded values) and `model_config.json`
- `Model_Training/`: architecture search, learning-rate comparison, training curves, threshold selection
- `Model_Evaluation/`: test metrics, confusion matrix, ROC/PR curves, calibration, feature importance, partial dependence, segment and gains charts
- `Predictions/churn_predictions.csv`: churn probability, prediction, risk tier and cluster for every customer (`row_id`). Only rows with `split = test` are an unbiased accuracy estimate

Input data is read from `../Data_Preparation/full_encoded_cleaned.csv` and `../Clustering_Analysis/cluster_assignments.csv`.

## Requirements
pandas, numpy, matplotlib, seaborn, scikit-learn (1.2+), joblib. No deep-learning framework needed.

## Loading the model
```python
import joblib, json, pandas as pd
model = joblib.load("Model/ann_churn_pipeline.joblib")
cfg = json.load(open("Model/model_config.json"))
X = pd.read_csv("../Data_Preparation/full_encoded_cleaned.csv")[cfg["features_in_order"]]
prob = model.predict_proba(X)[:, 1]
pred = (prob >= cfg["decision_threshold"]).astype(int)
```
