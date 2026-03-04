import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Show Streamlit version
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🛡️ Credit Card Fraud Detector")
st.info(f"Streamlit v{st.__version__} | Powered by TensorFlow v{tf.__version__}")

# Load model & scaler
@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model("fraud_model.h5")  # Or fraud_model.keras
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Missing files! Need `fraud_model.h5` and `scaler.pkl` in folder.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Load error: {e}")
        st.stop()

model, scaler = load_artifacts()
st.success("✅ Model & scaler loaded!")

# Features (29 total: V1-V28 + Amount)
features = [f'V{i}' for i in range(1, 29)] + ['Amount']

# Inputs (sidebar)
st.sidebar.header("💳 Enter Transaction Data")
input_data = {}
for feat in features[:-1]:  # V1-V28 sliders
    input_data[feat] = st.sidebar.slider(feat, -35.0, 35.0, 0.0, 0.1)
amount = st.sidebar.number_input("Amount ($)", 0.0, 50000.0, 88.35)
input_data['Amount'] = amount

# Predict
if st.sidebar.button("🚨 Check for Fraud", type="primary"):
    df = pd.DataFrame([input_data])
    
    # Scale Amount only
    df[['Amount']] = scaler.transform(df[['Amount']])
    
    # Predict
    prob = model.predict(df, verbose=0)[0][0]
    
    # Results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fraud Risk", f"{prob:.1%}")
    with col2:
        if prob > 0.5:
            st.error("🚨 **FRAUD DETECTED**")
        else:
            st.success("✅ **Legitimate**")
    with col3:
        st.info(f"Amount: ${amount:,.0f}")
    
    # Features chart
    st.subheader("📊 Feature Profile")
    st.bar_chart(df.T)

# Footer
st.markdown("---")
st.caption("Built with your Azure ML fraud model | Deployed via Streamlit Cloud")
