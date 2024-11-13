from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import xgboost as xgb

app = Flask(__name__, template_folder='template')

# Load model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgb_fraud_detection_model.json")

# Route for home page
@app.route('/')
def home():
    return render_template('index1.html')

# Route for form page
@app.route('/form', methods=['GET', 'POST'])
def form():
    return render_template('form.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    account_number = request.form['account_number']
    receiver_acc_number = request.form['receiver_acc_number']
    amount = float(request.form['amount'])
    Type = int(request.form['Type'])

    # Load datasets
    bank = pd.read_csv('Financial.csv')
    fraud_accounts = pd.read_csv('fraud_accounts.csv')
    
    # Fetch sender and receiver account info
    account_info_sender = bank[bank['nameOrig'] == account_number]
    account_info_receiver = bank[bank['nameDest'] == receiver_acc_number]

    # Check if receiver account is flagged as fraud
    if receiver_acc_number in fraud_accounts['account_number'].values:
        message = f"This account {receiver_acc_number} is already flagged as fraud."
        return render_template('fraud_flagged.html', message=message)

    # Ensure account details are available
    if not account_info_sender.empty and not account_info_receiver.empty:
        # Extract account balances
        oldbalanceOrg = account_info_sender['oldbalanceOrg'].values[0]
        newbalanceOrig = account_info_sender['newbalanceOrig'].values[0]
        oldbalanceDest = account_info_receiver['oldbalanceDest'].values[0]
        newbalanceDest = account_info_receiver['newbalanceDest'].values[0]
        
        # Prepare input features
        input_features = [Type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]
        input_array = np.array(input_features).reshape(1, -1)

        # Model prediction
        possibility_of_fraud = xgb_model.predict(input_array)[0]
        probability_of_fraud = xgb_model.predict_proba(input_array)[0][1]

        if possibility_of_fraud == 1:
            message = f"Account {receiver_acc_number} might be involved in fraud."
            proba = f"{probability_of_fraud*100:.2f}"
            return render_template('flag_account.html', proba=proba, message=message, account_number=receiver_acc_number, balance=oldbalanceDest)
        else:
            message = f"Transaction seems normal for account {receiver_acc_number}."
            proba = f"{probability_of_fraud*100:.2f}"
            return render_template('transaction_result.html', message=message, proba=proba)
    else:
        message = "The Account Number provided is invalid or not available in the database. Please try another account number."
        return render_template('Invalid.html', message=message)

# Route for flagging an account as fraud
@app.route('/flag', methods=['POST', 'GET'])
def flag():
    account_number = request.form.get('account_number')
    balance = str(request.form.get('balance'))
    flag_decision = request.form.get('flag')

    if flag_decision == "yes":
        # Update fraud accounts list
        fraud_accounts = pd.read_csv('fraud_accounts.csv')
        new_entry = pd.DataFrame([[account_number, balance]], columns=['account_number', 'balance'])
        fraud_accounts = pd.concat([fraud_accounts, new_entry], ignore_index=True)
        fraud_accounts.to_csv('fraud_accounts.csv', index=False)
        return f"Account {account_number} has been flagged as fraud."
    else:
        return f"Account {account_number} was not flagged as fraud."

if __name__ == '__main__':
    app.run(debug=True)
