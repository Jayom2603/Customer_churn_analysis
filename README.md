Customer Churn Prediction

An end-to-end machine-learning project that predicts whether a customer is likely to churn. It covers data cleaning, exploratory data analysis (EDA), preprocessing, XGBoost model development, evaluation, threshold selection, and an interactive Streamlit dashboard.

## Tech Stack

- Python
- Pandas, NumPy, Matplotlib
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

## Current Status

The XGBoost training and evaluation pipeline is complete. The final model is saved as `customer_churn_model.pkl` and can be used through the Streamlit dashboard to estimate churn probability for new customers.

## Workflow

```text
Raw customer data
        ↓
Data cleaning and EDA
        ↓
Feature preprocessing
        ↓
XGBoost classification model
        ↓
Held-out test evaluation and 5-fold stratified CV
        ↓
Threshold analysis
        ↓
Saved pipeline: customer_churn_model.pkl
        ↓
Streamlit dashboard
```

## Model Evaluation

The final XGBoost model was evaluated on a held-out test set.

| Metric | Score |
|---|---:|
| Accuracy | 95.26% |
| Precision | 87.80% |
| Recall | 80.90% |
| F1-score | 84.21% |
| ROC-AUC | ~98% |

The project also uses 5-fold Stratified Cross-Validation to assess performance stability.

### Threshold Selection

Classification thresholds from `0.10` to `0.90` were tested. A threshold of `0.50` was selected because it achieved the highest tested F1-score.

## Streamlit Dashboard

- **Overview** — customer and churn statistics
- **Churn Analysis** — churn patterns and customer-value analysis
- **Model Performance** — evaluation metrics
- **New Customer Prediction** — churn-probability prediction for a new customer

## Project Structure

```text
.
├── data/                       # Input dataset(s)
├── notebooks/                  # EDA and experimentation
├── customer_churn_model.pkl    # Saved trained pipeline
├── app.py                      # Streamlit application
├── requirements.txt            # Project dependencies
└── README.md
```

## Setup and Run

```bash
git clone <your-repository-url>
cd <your-repository-folder>
pip install -r requirements.txt
streamlit run app.py
```

## Conclusion

This project provides a complete churn-prediction workflow, from data preparation through dashboard-based predictions, with XGBoost as the final model and `0.50` as the selected decision threshold.
