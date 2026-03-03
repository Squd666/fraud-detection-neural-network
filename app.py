import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

# 1. SETUP
st.set_page_config(page_title="Sentinel AI", page_icon="🛡️")
st.title("🛡️ Project Sentinel")

# 2. LOAD ASSETS
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('fraud_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# 3. INPUTS
st.subheader("Transaction Analysis")
amount = st.number_input("Transaction Amount ($)", value=125.0)

# 4. PREDICTION LOGIC
if st.button("Run Fraud Analysis"):
    # We must satisfy the scaler's 30-feature requirement
    # Even if we use zeros, the shape must be correct: [Time, V1...V28, Amount]
    dummy_v = [0] * 28
    full_raw_data = [0] + dummy_v + [amount]
    
    # Step A: Scale all 30 features (matches scaler.pkl)
    scaled_data = scaler.transform([full_raw_data])
    
    # Step B: Remove 'Time' (index 0) to get the 29 features (matches fraud_model.keras)
    model_input = scaled_data[:, 1:]
    
    # Step C: Predict
    prediction = model.predict(model_input)
    score = float(prediction[0][0])
    
    st.divider()
    if score > 0.5:
        st.error(f"🚨 FRAUD ALERT: {score:.2%}")
    else:
        st.success(f"✅ TRANSACTION CLEAR: {score:.2%}")
