import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

# Page setup
st.set_page_config(page_title="Sentinel AI", page_icon="🛡️")
st.title("Project Sentinel")

# Load assets
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('fraud_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# Inputs
time = st.number_input("Time", value=0)
amount = st.number_input("Amount", value=100.0)

v_inputs = []
cols = st.columns(4)
for i in range(1, 29):
    with cols[(i-1) % 4]:
        val = st.number_input(f"V{i}", value=0.0)
        v_inputs.append(val)

# Prediction Logic
if st.button("Analyze"):
    # 1. Scaler expects 30 features: Time + V1-V28 + Amount
    full_features = [time] + v_inputs + [amount]
    input_array = np.array([full_features])
    
    # 2. Transform (Scales all 30)
    scaled_data = scaler.transform(input_array)
    
    # 3. Model expects 29 features (We drop Time, which is index 0)
    model_input = scaled_data[:, 1:]
    
    # 4. Predict
    prediction = model.predict(model_input)
    score = prediction[0][0]
    
    if score > 0.5:
        st.error(f"Fraud Detected: {score:.2%}")
    else:
        st.success(f"Transaction Secure: {score:.2%}")
