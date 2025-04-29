# Financial Fraud Detection using XGBoost

This project is a robust machine learning solution for detecting fraudulent transactions in a financial dataset. Using upsampling techniques to balance the data and powerful classification via XGBoost, it achieves near-perfect accuracy in predicting fraudulent activity.

---

## Overview

- Predicts fraud based on transaction type, amount, sender/receiver balances.
- Utilizes XGBoost for high-performance classification.
- Provides functionality to **flag suspicious accounts** and track them.
- Real-time **interactive input** for testing new transactions.
- Achieved **99.95%** accuracy on test data.

---

## Dataset

- **Source**: `Financial.csv`
- **Features Used**:
  - `type` (CASH_IN, CASH_OUT, DEBIT, etc.)
  - `amount`
  - `oldbalanceOrg`, `newbalanceOrig`
  - `oldbalanceDest`, `newbalanceDest`
- **Target**: `isFraud` (0 or 1)

---

## Process

1. **Data Cleaning**:
   - Dropped non-essential columns like `nameOrig`, `nameDest`, `step`, `isFlaggedFraud`.

2. **Label Encoding**:
   - Encoded `type` column using `LabelEncoder`.

3. **Balancing the Dataset**:
   - Used upsampling (`resample`) to balance fraud vs. non-fraud transactions.

4. **Model Training**:
   - XGBoost classifier with `train_test_split`.

5. **Evaluation**:
   - Accuracy: `99.95%`
   - Confusion Matrix & Classification Report included.

---

## Fraud Account Tracking

- A list of known fraudulent accounts is created in `fraud_accounts.csv`.
- Predict function checks if the account is already flagged.
- Prompts the user if a new fraud is detected to flag it interactively.

---

## Requirements

```bash
pip install pandas numpy xgboost scikit-learn imblearn 
```

## Running the Project

1. Train the Model

Loads and processes data

Upsamples for balance

Trains XGBoost model

Evaluates accuracy and confusion matrix

2. Predict a Transaction (Interactive)

You’ll be prompted to enter:

Sender and receiver account numbers

Amount

Type of transaction

##  Example Prediction Flow

```bash
Enter sender's account number: C123456789
Enter receiver's account number: C987654321
Enter the amount to pay: 3500
Enter type of transaction 'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4: 1

Input Features:  [1, 3500.0, 12000.0, 8500.0, 5000.0, 8500.0]
Transaction seems normal for account C987654321.
Fraudulent Transaction Prediction: False
```

## Model Performance

```bash 
Metric	Score
Accuracy	99.95%
Precision	1.00
Recall	1.00
F1-Score	1.00

#Confusion Matrix:

[[246365    250]
 [     0 247866]]
```


## Future Work


Web dashboard for transaction simulation using Streamlit.

Integration with live bank APIs (mock).

Time-series based anomaly detection.

Alert system with fraud risk scoring.



