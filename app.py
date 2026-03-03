import streamlit as st
import tensorflow as tf
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Sentinel AI: Fraud Detector", page_icon="🛡️")
st.title("🛡️ Project Sentinel")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('fraud_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.success("✅ AI Engine Loaded Successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- DYNAMIC INPUTS ---
st.subheader("Transaction Details")
amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=125.0)

# We create an expander so the 28 features don't clutter the screen
with st.expander("Adjust Behavioral Features (V1-V28)"):
    v_inputs = []
    cols = st.columns(4)
    for i in range(1, 29):
        with cols[i % 4]:
            val = st.number_input(f"V{i}", value=0.0, step=0.1)
            v_inputs.append(val)

if st.button("Run Fraud Analysis"):
    try:
        # 1. Scale the amount
        scaled_amount = scaler.transform([[amount]])[0][0]
        
        # 2. Combine into a flat list: [V1...V28, Scaled_Amount]
        # This creates exactly 29 features
        final_features = v_inputs + [scaled_amount]
        
        # 3. Reshape for TensorFlow: (1, 29)
        final_input = np.array([final_features])
        
        # 4. Predict
        prediction = model.predict(final_input)
        prob = prediction[0][0]
        
        if prob > 0.5:
            st.error(f"🚨 ALERT: HIGH FRAUD RISK ({prob:.2%})")
        else:
            st.success(f"✅ TRANSACTION SAFE ({prob:.2%})")
            
    except Exception as e:
        st.error(f"Input Error: {e}. Check if your model expects 29 or 30 features.")
