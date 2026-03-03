import streamlit as st
import tensorflow as tf
import pandas as pd
import joblib
import numpy as np
import datetime

# Basic page setup
st.set_page_config(page_title="Sentinel AI: Fraud Detector", page_icon="🛡️")
st.title("Project Sentinel")

# Initialize the activity log in the background
if 'activity_log' not in st.session_state:
    st.session_state.activity_log = []

# Load the AI model and scaler once to keep things fast
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('fraud_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.success("AI Engine is ready")
except Exception as e:
    st.error(f"Could not load the model: {e}")

# Main user input area
st.header("Transaction Analysis")
amount = st.number_input("How much is the transaction? ($)", min_value=0.01, value=125.00)
v14_input = st.number_input("Behavioral Feature V14 (Test -10 for Fraud)", value=0.0)

# Process the prediction when the user clicks the button
if st.button("Check for Fraud"):
    # Prepare the input data
    # We use 0.0 for most features but plug in your V14 value
    v_features = np.zeros((1, 28)) 
    v_features[0, 13] = v14_input
    
    # Standardize the amount using our saved scaler
    scaled_amount = scaler.transform([[amount]])
    
    # Combine everything to match the 29 inputs the model expects
    final_input = np.hstack([v_features, scaled_amount])
    
    # Run the prediction
    prediction = model.predict(final_input)
    fraud_probability = prediction[0][0]
    
    # Log this event for monitoring purposes
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    status = "FRAUD" if fraud_probability > 0.5 else "SAFE"
    log_entry = f"Time: {timestamp} | Amount: ${amount} | V14: {v14_input} | Score: {fraud_probability:.2%} | Result: {status}"
    st.session_state.activity_log.append(log_entry)

    # Display the result to the user
    if fraud_probability > 0.5:
        st.error(f"High risk of fraud detected! ({fraud_probability:.2%})")
    else:
        st.success(f"This transaction looks safe ({fraud_probability:.2%})")

# The Monitoring Section
# This acts as your diary to track how the model is performing
if st.session_state.activity_log:
    st.markdown("---")
    st.subheader("System Monitoring Log")
    for entry in reversed(st.session_state.activity_log):
        st.text(entry)

# Simple sidebar info
st.sidebar.title("System Info")
st.sidebar.write("Model: Neural Network")
st.sidebar.write("Recall Rate: 82%")
