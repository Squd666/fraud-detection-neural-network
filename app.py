import streamlit as st
import tensorflow as tf
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Sentinel AI", page_icon="🛡️")
st.title("Project Sentinel")
st.write("Neural Network Fraud Detection")

# Load assets
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('fraud_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# Transaction Inputs
st.header("Transaction Analysis")
time = st.number_input("Time (Seconds)", value=0)
amount = st.number_input("Amount ($)", value=100.0)

with st.expander("Behavioral Features (V1-V28)"):
    v_inputs = []
    cols = st.columns(4)
    # Your notebook uses exactly V1 through V28
    for i in range(1, 29):
        with cols[(i-1) % 4]:
            val = st.number_input(f"V{i}", value=0.0)
            v_inputs.append(val)

if st.button("Run Prediction"):
    # Step 1: Create the feature list in the EXACT training order
    # Your notebook order: [Time, V1, V2, ..., V28, Amount]
    feature_list = [time] + v_inputs + [amount]
    
    # Step 2: Convert to a format the Scaler understands (a 2D Array)
    input_data = np.array([feature_list])
    
    # Step 3: Transform using the Scaler
    # This matches the 30-feature requirement from your notebook
    try:
        scaled_data = scaler.transform(input_data)
        
        # Step 4: Predict
        prediction = model.predict(scaled_data)
        risk_score = prediction[0][0]
        
        st.divider()
        if risk_score > 0.5:
            st.error(f"HIGH RISK DETECTED: {risk_score:.2%}")
        else:
            st.success(f"TRANSACTION SECURE: {risk_score:.2%}")
            
    except ValueError as e:
        st.error("Feature Mismatch Error")
        st.info("The model expects 30 features. Check your V1-V28 inputs.")

st.sidebar.info("Model Status: Online\nExpected Features: 30")
