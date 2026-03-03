import streamlit as st
import tensorflow as tf
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Sentinel AI: Fraud Detector", page_icon="🛡️")
st.title("Project Sentinel")
st.write("This tool uses a neural network to check transactions for fraudulent patterns.")

# Load the model and scaler
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('fraud_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.success("AI Engine is online")
except Exception as e:
    st.error(f"Could not load assets: {e}")

# User inputs
st.header("Transaction Analysis")
# Your notebook includes 'Time' as the first feature
time = st.number_input("Time (Seconds since first transaction)", min_value=0, value=0)
amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=125.00)

# Organized grid for features V1 through V28
with st.expander("Adjust Behavioral Features (V1-V28)"):
    st.write("Modify these values to test how the model responds.")
    v_inputs = []
    cols = st.columns(4)
    
    # We loop from 1 to 28 because V29 doesn't exist in your training set
    for i in range(1, 29):
        col_index = (i - 1) % 4
        with cols[col_index]:
            val = st.number_input(f"V{i}", value=0.0, step=0.1)
            v_inputs.append(val)

# Run the analysis
if st.button("Analyze Transaction"):
    # Step 1: Scale the amount
    scaled_amount = scaler.transform([[amount]])[0][0]
    
    # Step 2: Combine in exact training order: [Time, V1...V28, Scaled Amount]
    # This creates the 30 features your model expects
    final_features = [time] + v_inputs + [scaled_amount]
    final_input = np.array([final_features])
    
    # Step 3: Get prediction
    prediction = model.predict(final_input)
    fraud_probability = prediction[0][0]
    
    st.markdown("---")
    if fraud_probability > 0.5:
        st.error(f"High risk detected: {fraud_probability:.2%}")
        st.warning("This transaction matches patterns often associated with fraud.")
    else:
        st.success(f"Transaction cleared: {fraud_probability:.2%}")
        st.write("The AI did not find significant risk factors.")

# Sidebar status
st.sidebar.title("System Info")
st.sidebar.write("Model: Deep Neural Network")
st.sidebar.write("Input Size: 30 Features")
