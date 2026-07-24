import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Model
# ==========================
model = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ==========================
# Title
# ==========================
st.title("🏦 Loan Approval Prediction System")

st.write("Enter Applicant Details")

# ==========================
# Input Fields
# ==========================

no_of_dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=20,
    value=1
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0,
    value=500000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=200000
)

loan_term = st.number_input(
    "Loan Term (Years)",
    min_value=1,
    value=20
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=750
)

residential_assets_value = st.number_input(
    "Residential Assets Value",
    min_value=0,
    value=500000
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value",
    min_value=0,
    value=200000
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value",
    min_value=0,
    value=200000
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0,
    value=100000
)

# ==========================
# Encoding
# ==========================

education_encoded = 0 if education == "Graduate" else 1
self_employed_encoded = 1 if self_employed == "Yes" else 0

# ==========================
# Prediction
# ==========================

if st.button("Predict Loan Status"):

    input_df = pd.DataFrame([[
        no_of_dependents,
        education_encoded,
        self_employed_encoded,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]], columns=columns)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.subheader("Input Details")

    st.write("Dependents :", no_of_dependents)
    st.write("Education :", education)
    st.write("Self Employed :", self_employed)
    st.write("Annual Income :", income_annum)
    st.write("Loan Amount :", loan_amount)
    st.write("Loan Term :", loan_term)
    st.write("CIBIL Score :", cibil_score)
    st.write("Residential Assets :", residential_assets_value)
    st.write("Commercial Assets :", commercial_assets_value)
    st.write("Luxury Assets :", luxury_assets_value)
    st.write("Bank Assets :", bank_asset_value)