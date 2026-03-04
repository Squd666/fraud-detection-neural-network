import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# -------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="Sentinel AI: Fraud Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Project Sentinel")
st.markdown("### Neural Network Fraud Detection System")
st.write("This AI model analyzes transactions to identify high-risk fraudulent patterns.")

# -------------------------------------------------
# 2. LOAD MODEL & SCALER (Cached)
# -------------------------------------------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("fraud_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
    st.success("✅ AI Engine Loaded Successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------------------------------
# 3. USER INPUT SECTION
# -------------------------------------------------
st.divider()
st.subheader("Transaction Details")

st.info("Model expects 28 PCA-transformed features (V1–V28) plus Scaled Amount.")

# Collect V1–V28 inputs
v_features = []

for i in range(1, 29):
    value = st.number_input(f"V{i}", value=0.0, format="%.6f")
    v_features.append(value)

# Transaction Amount (to be scaled)
amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=125.00)

# -------------------------------------------------
# 4. PREDICTION LOGIC
# -------------------------------------------------
if st.button("Run Fraud Analysis"):

    try:
        # Convert V-features to correct shape (1, 28)
        v_array = np.array(v_features).reshape(1, 28)

        # Scale Amount exactly as in training
        scaled_amount = scaler.transform([[amount]])  # shape (1,1)

        # Combine into final input (1, 29)
        final_input = np.hstack([v_array, scaled_amount])

        # Safety check
        if final_input.shape[1] != 29:
            st.error(f"Model expects 29 features, but received {final_input.shape[1]}")
            st.stop()

        # Predict
        prediction = model.predict(final_input)
        fraud_probability = float(prediction[0][0])

        # -------------------------------------------------
        # 5. DISPLAY RESULTS
        # -------------------------------------------------
        st.divider()

        if fraud_probability >= 0.5:
            st.error("🚨 HIGH FRAUD RISK DETECTED")
            st.metric("Fraud Probability", f"{fraud_probability:.2%}")
            st.warning("Recommended Action: Decline transaction and notify cardholder.")
        else:
            st.success("✅ TRANSACTION CLEARED")
            st.metric("Fraud Probability", f"{fraud_probability:.2%}")
            st.write("Pattern consistent with legitimate transaction.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# -------------------------------------------------
# 6. FOOTER
# -------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.write("Built with **TensorFlow** & **Azure ML**")
st.sidebar.write("Model Input Shape: (None, 29)")
