import streamlit as st
import tensorflow as tf
import pandas as pd
import joblib
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Sentinel AI: Fraud Detector", page_icon="🛡️")
st.title("🛡️ Project Sentinel")
st.markdown("### Neural Network Fraud Detection System")
st.write("This AI model analyzes transactions to identify high-risk fraudulent patterns.")

# --- 2. LOAD TRAINED MODEL & SCALER ---
@st.cache_resource  # This keeps the app fast by loading the model only once
def load_assets():
    # Loading the native Keras format as recommended by TensorFlow
    model = tf.keras.models.load_model('fraud_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.success("✅ AI Engine Loaded Successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- 3. USER INPUT INTERFACE ---
st.divider()
st.subheader("Transaction Details")

# In a real bank, these 'V' features are anonymized telemetry data.
# We will allow the user to input the Amount and simulate the rest.
amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=125.00)

st.info("💡 Technical Note: This model processes 28 PCA-transformed 'V' features plus the scaled Amount.")

# --- 4. PREDICTION LOGIC ---
if st.button("Run Fraud Analysis"):
    # Preprocessing: The model expects 30 features total (V1-V28 + V29 + Scaled Amount)
    # We create a dummy array for V1-V29 for demonstration purposes
    dummy_v_features = np.zeros((1, 29)) 
    
    # Scale the amount exactly as we did during training
    scaled_amount = scaler.transform([[amount]])
    
    # Combine features into the final input shape (1, 30)
    final_input = np.hstack([dummy_v_features, scaled_amount])
    
    # Make Prediction
    prediction = model.predict(final_input)
    fraud_probability = prediction[0][0]
    
    # Display Results
    st.divider()
    if fraud_probability > 0.5:
        st.error(f"🚨 ALERT: HIGH FRAUD RISK DETECTED")
        st.metric("Fraud Probability", f"{fraud_probability:.2%}")
        st.warning("Recommended Action: Decline transaction and notify cardholder.")
    else:
        st.success(f"✅ TRANSACTION CLEARED")
        st.metric("Fraud Probability", f"{fraud_probability:.2%}")
        st.write("Confidence: Legitimate transaction pattern identified.")

# --- 5. FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("Built with **TensorFlow** & **Azure ML**")
st.sidebar.write("Target Recall: **82%**")