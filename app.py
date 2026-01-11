import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Insurance Claim Risk Predictor", layout="centered")

st.title("🚗 Insurance Claim Risk Prediction")
st.write("Predict the probability of an insurance claim using a trained ML model.")

# -----------------------------
# LOAD MODEL & FEATURES
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    features = joblib.load("feature_columns.pkl")
    return model, features

model, feature_columns = load_model()

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("Enter Policy & Vehicle Details")

user_input = {}

for feature in feature_columns:
    user_input[feature] = st.number_input(
        label=feature,
        value=0.0
    )

input_df = pd.DataFrame([user_input])

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict Claim Risk"):
    prob = model.predict_proba(input_df)[0][1]

    st.metric("📊 Claim Probability", f"{prob:.2%}")

    if prob > 0.6:
        st.error("🔴 High Risk Policy")
    elif prob > 0.3:
        st.warning("🟠 Medium Risk Policy")
    else:
        st.success("🟢 Low Risk Policy")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built with Streamlit | Insurance Claim ML Model")
