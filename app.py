import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

# Page setup
st.set_page_config(page_title="Sentinel AI", page_icon="🛡️")
st.title("Project Sentinel")
st.write("Neural Network Fraud Detection System")

# Load assets
@st.cache_resource
def load_assets():
    # Load the model and the scaler precisely as named in your notebook
    model = tf.keras.models.load_model('fraud_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.success("AI Engine Online")
except Exception as e:
    st.error(f"Asset Error: {e}")

# Inputs (Creating the 30 features the model expects)
st.header("Transaction Details")
time = st.number_input("Time (Seconds)", value=0)
amount = st.number_input("Transaction Amount ($)", value=100.0)

with st.expander("Behavioral Features (V1-V28)"):
    v_inputs = []
    cols = st.columns(4)
    for i in range(1, 29):
        with cols[(i-1) % 4]:
            val = st.number_input(f"V{i}", value=0.0)
            v_inputs.append(val)

if st.button("Analyze"):
    # 1. Create the feature list in the EXACT training order: 
    # [Time, V1...V28, Amount]
    features = [time] + v_inputs + [amount]
    
    # 2. Convert to a 2D array for the scaler
    input_array = np.array([features])
    
    # 3. SCALE ALL 30 FEATURES AT ONCE
    # Your scaler was trained on all columns, so it must transform all columns.
    scaled_input = scaler.transform(input_array)
    
    # 4. Predict
    prediction = model.predict(scaled_input)
    score = prediction[0][0]
    
    st.divider()
    if score > 0.5:
        st.error(f"🚨 FRAUD DETECTED ({score:.2%})")
    else:
        st.success(f"✅ CLEAR ({score:.2%})")
