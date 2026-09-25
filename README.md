# Mobile Wallet Transaction Failure Prediction & Demand Forecasting

An end-to-end Python machine learning project built for **AWS Machine Learning Foundations (Variant 06)**. This repository contains a complete pipeline featuring binary classification to predict mobile wallet payment failures and time-series forecasting to project daily transaction volumes for a 7-day horizon.

---

## 📌 Project Overview

This project addresses two core operational challenges faced by mobile wallet platforms:
1. **Transaction Failure Risk (Classification):** Identifying high-risk transactions caused by network latency, low balances, or past failure history to reduce user churn and transaction drops.
2. **Transaction Demand Projections (Forecasting):** Forecasting short-term transaction volume to support server capacity planning and system load balancing.

---

## 🛠️ Key Deliverables & Workflow

### Part 1: Mobile Wallet Transaction Failure Prediction (Classification)
* **Exploratory Data Analysis (EDA):** Analyzed feature distributions, missing values, class balance, and inter-feature correlations using `seaborn` heatmaps and count plots.
* **Data Preprocessing & Cleaning:**
  * Imputed missing values via median (numerical) and mode (categorical).
  * Capped network latency outliers using the Interquartile Range (IQR) method.
  * Dropped non-predictive features (`Transaction_ID`) and applied one-hot encoding.
* **Model Training & Evaluation:**
  * Split data into reproducible stratified train/test sets (80/20).
  * Evaluated a **Random Forest Classifier** across Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
* **Threshold & Hyperparameter Optimization:**
  * Reduced the probability decision threshold from `0.50` to `0.35` to significantly improve sensitivity/recall on imbalanced failure events.
  * Tuned model parameters (`max_depth=5`, `min_samples_split=5`) to prevent decision tree overfitting.

### Part 2: Daily Transaction Volume Forecasting (Time-Series)
* **Chronological Preparation:** Converted timestamp features, sorted time-series observations, and handled missing values using linear interpolation.
* **Forecasting Pipeline:** Applied **Holt's Linear Exponential Smoothing** to project transaction counts across a **7-day forecast horizon**.
* **Visualization & Export:** Generated visual trend comparisons between historical performance (last 30 days) and projected demand, outputting a structured forecast table.


---

## 📂 Repository Structure

```text
├── P06_mobile_wallet_transaction_failure.csv   # Classification dataset
├── P06_daily_mobile_wallet_transactions.csv    # Forecasting dataset
├── Variant06_Mobile_Wallet_Analysis.ipynb       # Executable Python Jupyter Notebook
├── Variant06_Mobile_Wallet_Analysis.pdf         # Formatted PDF Report
└── README.md                                    # Project documentation
