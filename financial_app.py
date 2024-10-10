import streamlit as st 
import pandas as pd
import plotly.graph_objects as go
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.set_page_config(
                page_title="Financial Fraud",
                page_icon="💰",
                layout="wide",
                initial_sidebar_state="expanded"
                )
# Load model and scaler
dt_model = joblib.load("Decision_tree_smote.pkl")
encoder = joblib.load("onehot_encoder.pkl")  
def clean_data():
        df = pd.read_csv("new_financial.csv")
        return df

def add_predictions(input_data):
        # Convert input_data (currently a dictionary) to a DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Rename 'transaction_type' back to 'type' to match encoder's fit time
        input_df = input_df.rename(columns={'transaction_type': 'type'})
        
        # Apply OneHotEncoder to the 'type' column
        input_df_encoded = encoder.transform(input_df[['type']])
        
        # Convert the encoded array back into a DataFrame
        input_df_encoded = pd.DataFrame(input_df_encoded, columns=encoder.get_feature_names_out(['type']))
        
        # Drop the original 'type' column and combine the encoded columns
        input_df = input_df.drop(columns=['type'])
        input_df = pd.concat([input_df, input_df_encoded], axis=1)
        
        # Here we skip the scaling step
        # Make prediction using the pre-trained DecisionTree model
        prediction = dt_model.predict(input_df)
        
        # Return the prediction
        return "Fraud" if prediction[0] == 1 else "Not Fraud"


def get_radar_chart(data):
        categories = ['Transaction Amount', 'Old Balance of Origin', 'New Balance of Origin',
                        'Old Balance of Destination', 'New Balance of Destination']
        values = [data['amount'],
                data['oldbalanceOrg'],
                data['newbalanceOrig'],
                data['oldbalanceDest'],
                data['newbalanceDest']
                ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values,
                                        theta=categories,
                                        fill='toself',
                                        name='Selected Values'
                                        ))
        
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,
                                                range=[0, max(values)]
                                                )
                        ),
                        showlegend=True
        )
        return fig

def add_sidebar():
        data = clean_data()
        
        amount = st.sidebar.slider("Transaction Amount", 
                                min_value=float(data['amount'].min()), 
                                max_value=float(data['amount'].max()),
                                value=float(data['amount'].mean()),
                                key="amount_slider"
                                )
        
        oldbalanceOrg = st.sidebar.slider("Old Balance of Origin", 
                                min_value=float(data['oldbalanceOrg'].min()), 
                                max_value=float(data['oldbalanceOrg'].max()),
                                value=float(data['oldbalanceOrg'].mean()),
                                key="oldbalanceOrg_slider"
                                )
        
        newbalanceOrig = st.sidebar.slider("New Balance of Origin", 
                                min_value=float(data['newbalanceOrig'].min()), 
                                max_value=float(data['newbalanceOrig'].max()), 
                                value=float(data['newbalanceOrig'].mean()),
                                key="newbalanceOrig_slider"
                                )
        
        oldbalanceDest = st.sidebar.slider("Old Balance of Destination", 
                                min_value=float(data['oldbalanceDest'].min()), 
                                max_value=float(data['oldbalanceDest'].max()), 
                                value=float(data['oldbalanceDest'].mean()),
                                key="oldbalanceDest_slider"
                                )
        
        newbalanceDest = st.sidebar.slider("New Balance of Destination", 
                                min_value=float(data['newbalanceDest'].min()), 
                                max_value=float(data['newbalanceDest'].max()), 
                                value=float(data['newbalanceDest'].mean()),
                                key="newbalanceDest_slider"
                                )
        
        # Dropdowns for transaction details
        nameOrig_frq = st.sidebar.selectbox("Sender Frequency", sorted(data['nameOrig_frq'].unique()))
        nameDest_frq = st.sidebar.selectbox("Receiver Frequency", sorted(data['nameDest_frq'].unique()))
        
        transaction_type = st.sidebar.selectbox("Transaction Type",
                                                options=["CASH IN", "CASH OUT", "DEBIT", "PAYMENT", "TRANSFER"],
                                                index=0,
                                                key="type_select"
                                                )
        
        return  {"amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "nameOrig_frq": nameOrig_frq,
                "nameDest_frq": nameDest_frq,
                "transaction_type": transaction_type
                }

def add_footer():
        st.markdown("Copyright 2024. All rights reserved. Made by Nimmala Sowmya and Angel Maria Stanley")

def main():
        st.sidebar.header("Finance Amount")
        
        with st.container():
                st.title("Financial Fraud Prediction")
                st.write("This app helps you identify if you're sending the money to an account which is fraud or not.")
        
        input_data = add_sidebar()
        
        col1, col2 = st.columns([4, 1])
        
        radar_chart_data = {'amount': input_data['amount'],
                        'oldbalanceOrg': input_data['oldbalanceOrg'],
                        'newbalanceOrig': input_data['newbalanceOrig'],
                        'oldbalanceDest': input_data['oldbalanceDest'],
                        'newbalanceDest': input_data['newbalanceDest'],
                        'nameOrig_frq': input_data['nameOrig_frq'],
                        'nameDest_frq': input_data['nameDest_frq'],
                        'transaction_type': input_data['transaction_type']
                        }
        radar_chart = get_radar_chart(radar_chart_data)
        
        with col1:
                st.plotly_chart(radar_chart)
        
        with col2:
                prediction_result = add_predictions(input_data)
                st.write(f"**Prediction Result:** {prediction_result}")

if __name__ == "__main__":
        main()
