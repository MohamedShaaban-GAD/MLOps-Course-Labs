# Bank Customer Churn Prediction (MLflow + Scikit-learn)

This project demonstrates an end-to-end **machine learning workflow** for predicting customer churn in a banking dataset, with full **experiment tracking using MLflow**.

The goal is not only to train models, but to **compare multiple algorithms**, track metrics, log datasets, preprocessors, and artifacts in a reproducible MLOps-style setup.

---

## 📌 Project Features

- Data preprocessing with:
  - Class rebalancing
  - Numerical scaling
  - Categorical encoding
- Multiple ML models comparison:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVC)
- Experiment tracking with **MLflow**:
  - Parameters
  - Metrics
  - Models
  - Datasets
  - Preprocessing pipeline
  - Confusion matrix plots

---

## 📂 Project Structure

MLOps-Course-Labs/
│
├── dataset/
│ └── Churn_Modelling.csv
│
├── src/
│ └── train.py
│
├── churn_prediction/
│ └── Scripts/
│
├── mlruns/ # Created automatically by MLflow
│
└── README.md


---

## ⚙️ Requirements

- Python 3.9+
- MLflow
- Pandas
- Scikit-learn
- Matplotlib

Install dependencies:

```bash
pip install mlflow pandas scikit-learn matplotlib
```
---

