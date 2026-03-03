import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

# --- 1. SETUP ---
st.set_page_config(page_title="Sentinel AI", page_icon="🛡️")
st.title("🛡️ Project Sentinel")

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Matches your uploaded files
    model = tf.keras.models.load_model('fraud_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# --- 3. INPUTS ---
# Your scaler expects 30 features (Time + V1-V28 + Amount)
st.subheader("Transaction Analysis")
time = st.number_input("Time (Seconds)", value=0)
amount = st.number_input("Transaction Amount ($)", value=125.0)

# We must provide all V-features so the scaler doesn't crash
with st.expander("Behavioral Features (V1-V28)"):
    v_inputs = []
    cols = st.columns(4)
    for i in range(1, 29):
        with cols[(i-1) % 4]:
            val = st.number_input(f"V{i}", value=0.0)
            v_inputs.append(val)

# --- 4. PREDICTION LOGIC ---
if st.button("Run Fraud Analysis"):
    # Create the 30-feature list for the scaler: [Time, V1...V28, Amount]
    full_features = [time] + v_inputs + [amount]
    
    # Scale all 30 features (This satisfies scaler.pkl)
    scaled_data = scaler.transform([full_features])
    
    # Drop the first column (Time) to get the 29 features for the model
    # (This satisfies fraud_model.keras)
    model_input = scaled_data[:, 1:]
    
    # Make Prediction
    prediction = model.predict(model_input)
    score = float(prediction[0][0])
    
    st.divider()
    if score > 0.5:
        st.error(f"🚨 HIGH RISK: {score:.2%}")
    else:
        st.success(f"✅ CLEAR: {score:.2%}")
