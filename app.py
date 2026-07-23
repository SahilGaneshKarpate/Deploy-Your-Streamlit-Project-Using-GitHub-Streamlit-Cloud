import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Model Files
# ==========================
model = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")

st.title("🏦 Loan Approval Prediction System")
st.write("Enter Applicant Details")

# ==========================
# User Input
# ==========================

loan_id = st.number_input("Loan ID", min_value=1, step=1)

no_of_dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    step=1
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
    min_value=0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0
)

loan_term = st.number_input(
    "Loan Term (Years)",
    min_value=1
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900
)

residential_assets_value = st.number_input(
    "Residential Assets Value",
    min_value=0
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value",
    min_value=0
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value",
    min_value=0
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0
)

# ==========================
# Manual Encoding
# ==========================

education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0

# ==========================
# Prediction
# ==========================

if st.button("Predict Loan Status"):

    input_df = pd.DataFrame([[
        loan_id,
        no_of_dependents,
        education,
        self_employed,
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

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.subheader("Input Details")

    st.write("Loan ID :", loan_id)
    st.write("Dependents :", no_of_dependents)
    st.write("Education :", "Graduate" if education == 0 else "Not Graduate")
    st.write("Self Employed :", "Yes" if self_employed == 1 else "No")
    st.write("Annual Income :", income_annum)
    st.write("Loan Amount :", loan_amount)
    st.write("Loan Term :", loan_term)
    st.write("CIBIL Score :", cibil_score)
    st.write("Residential Assets :", residential_assets_value)
    st.write("Commercial Assets :", commercial_assets_value)
    st.write("Luxury Assets :", luxury_assets_value)
    st.write("Bank Assets :", bank_asset_value)