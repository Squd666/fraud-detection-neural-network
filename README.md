🛡️ Project Sentinel: Neural Network Fraud Detection
A production-ready deep learning system for identifying fraudulent credit card transactions.

🚀 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fraud-detection-neural-network-lqsjqccztdhcicryt6ectm.streamlit.app)

📊 Project Overview
This project is an end-to-end AI Engineering Pipeline designed to identify high-risk fraudulent patterns. It uses a Deep Neural Network trained on PCA-transformed behavioral features and transaction amounts.

🛠️ Key Features
* Neural Network Inference: Processes 29 unique features (V1-V28 + Scaled Amount) in real-time.
* Data Pipeline: Implements automatic feature scaling using a pre-trained `StandardScaler`.
* Production UI: A clean, interactive Streamlit interface for manual transaction testing.

🧪 How to Test
1. Normal Transaction: Leave all V-features at `0.0`. The model should return "Safe."
2. Fraudulent Pattern: Adjust V14 to `-5.0`. As seen in my testing, this typically triggers a ~76% Fraud Risk alert.

📈 Model Performance
* Target Recall: 82% (Optimized to minimize "False Negatives" in fraud detection).
* Architecture: Multi-layer Perceptron (MLP) built with TensorFlow/Keras.
