import streamlit as st
import tensorflow as tf
import pandas as pd
import joblib
import numpy as np
import datetime

# Basic page setup
st.set_page_config(page_title="Sentinel AI: Fraud Detector", page_icon="🛡️")
st.title("Project Sentinel")

# Initialize the activity log to track model performance
if 'activity_log' not in st.session_state:
    st.session_state.activity_log = []

# Load the AI model and scaler
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('fraud_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.success("AI Engine is ready")
except Exception as e:
    st.error(f"Could not load the model assets: {e}")

st.header("Transaction Analysis")

# Main visible inputs
amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=125.00)

# Hidden but accessible features (V1-V28)
# We put these in an expander so the app stays clean for your portfolio
with st.expander("Adjust Behavioral Features (V1-V28)"):
    st.write("These are the PCA-transformed features the model uses to detect patterns.")
    v_inputs = []
    # Creating 4 columns to make the 28 inputs look organized
    cols = st.columns(4)
    for i in range(1, 29):
        with cols[i % 4]:
            # Each V-feature defaults to 0.0 (normal behavior)
            val = st.number_input(f"V{i}", value=0.0, step=0.1)
            v_inputs.append(val)

# Run the prediction logic
if st.button("Check for Fraud"):
    # 1. Scale the amount to match training data
    scaled_amount = scaler.transform([[amount]])[0][0]
    
    # 2. Combine the 28 V-features and the 1 scaled amount into one list
    # This creates the (1, 29) input shape your model requires
    final_features = v_inputs + [scaled_amount]
    final_input = np.array([final_features])
    
    # 3. Get the prediction score
    prediction = model.predict(final_input)
    fraud_probability = prediction[0][0]
    
    # 4. Log the result for Monitoring
    # We record the time, the risk score, and the most important feature (V14)
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    status = "FRAUD" if fraud_probability > 0.5 else "SAFE"
    v14_val = v_inputs[13] # V14 is at index 13
    
    log_entry = {
        "Time": timestamp,
        "V14": v14_val,
        "Score": f"{fraud_probability:.2%}",
        "Result": status
    }
    st.session_state.activity_log.append(log_entry)

    # 5. Display results
    if fraud_probability > 0.5:
        st.error(f"High risk of fraud detected! ({fraud_probability:.2%})")
    else:
        st.success(f"This transaction looks safe ({fraud_probability:.2%})")

# The Monitoring Dashboard
# This shows you a history of your tests so you can see the model "drifting"
if st.session_state.activity_log:
    st.markdown("---")
    st.subheader("Production Monitoring Log")
    st.write("Below is the history of every transaction analyzed in this session.")
    # Display the logs as a nice table
    log_df = pd.DataFrame(st.session_state.activity_log)
    st.table(log_df.iloc[::-1]) # Show latest on top

st.sidebar.title("System Info")
st.sidebar.write("Model: Neural Network")
st.sidebar.write("Status: Monitoring Active")
