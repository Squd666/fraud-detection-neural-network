import streamlit as st
import tensorflow as tf
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Sentinel AI", page_icon="🛡️")
st.title("Project Sentinel")
st.write("Neural Network Fraud Detection System")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('fraud_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.header("Transaction Details")
time = st.number_input("Time (Seconds)", value=0)
amount = st.number_input("Amount ($)", value=100.0)

with st.expander("Behavioral Features (V1-V28)"):
    v_inputs = []
    cols = st.columns(4)
    for i in range(1, 29):
        with cols[(i-1) % 4]:
            val = st.number_input(f"V{i}", value=0.0)
            v_inputs.append(val)

if st.button("Analyze Transaction"):
    # 1. Create the 30-feature list for the SCALER
    # Order: Time, V1-V28, Amount
    full_feature_list = [time] + v_inputs + [amount]
    
    # 2. Scale all 30 features
    input_data = np.array([full_feature_list])
    scaled_data = scaler.transform(input_data)
    
    # 3. DROP the 'Time' column (index 0) because the MODEL only wants 29 features
    # This is the secret step that stops the crash
    model_input = scaled_data[:, 1:] 
    
    # 4. Predict
    prediction = model.predict(model_input)
    fraud_chance = prediction[0][0]
    
    st.divider()
    if fraud_chance > 0.5:
        st.error(f"HIGH RISK: {fraud_chance:.2%}")
    else:
        st.success(f"SECURE: {fraud_chance:.2%}")

st.sidebar.write("Model Input: 29 Features")
st.sidebar.write("Scaler Input: 30 Features")
