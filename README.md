# Anomaly Detection (Imbalanced Classification) with MLflow Tracking

This project demonstrates an end-to-end applied ML workflow for detecting rare events (anomaly detection framed as **imbalanced binary classification**). It includes data simulation, model training, imbalance handling, evaluation, and **experiment tracking + model registry** using MLflow.

> Focus: practical ML pipeline skills (model iteration, evaluation discipline, experiment tracking, and reproducibility).

---

## What This Project Covers

### 1) Data Preparation
- Generates a **synthetic imbalanced dataset** using `sklearn.datasets.make_classification`
- Uses stratified train/test split to preserve class distribution

### 2) Baseline + Model Comparison
Trains and evaluates multiple classifiers:
- **Logistic Regression**
- **Random Forest**
- **XGBoost**

Evaluation uses `classification_report` (precision/recall/F1), with attention to minority-class performance.

### 3) Handling Class Imbalance
Applies **SMOTETomek** (oversampling + cleaning) to improve recall on the minority/anomaly class, then retrains XGBoost.

### 4) Experiment Tracking with MLflow
Tracks:
- model hyperparameters
- evaluation metrics (accuracy, recall by class, macro F1)
- trained models (sklearn/xgboost logging)

Runs are logged under an MLflow experiment for repeatable comparison.

### 5) Model Registry (Basic MLOps Workflow)
- Registers the best-performing model in MLflow Model Registry
- Demonstrates loading the model by name/version
- Copies a model version into a “production” registry entry (example promotion flow)

---

## Tech Stack
- Python
- NumPy, scikit-learn
- XGBoost
- imbalanced-learn (`SMOTETomek`)
- MLflow (tracking + model registry)

---

## How to Run

### 1) Install dependencies
```bash
pip install numpy scikit-learn xgboost imbalanced-learn mlflow
