# 🏦 Loan Approval Prediction System

A Machine Learning based Loan Approval Prediction application developed using Python, Scikit-learn and Streamlit.

## Project Overview

This project predicts whether a loan application is likely to be:

- Approved
- Rejected

The Machine Learning model used in this project is Logistic Regression.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Selection
5. Label Encoding
6. Feature Scaling
7. Train-Test Split
8. Logistic Regression
9. Model Evaluation
10. Streamlit Application
11. GitHub Deployment

## Project Files

- `app.py` - Streamlit application
- `loan_approval_model.pkl` - Trained Logistic Regression model
- `scaler.pkl` - StandardScaler object
- `columns.pkl` - Feature column names
- `loan_approval_dataset.csv` - Dataset
- `requirements.txt` - Required Python libraries
- `README.md` - Project documentation

## Features Used

The model uses 11 features:

1. no_of_dependents
2. education
3. self_employed
4. income_annum
5. loan_amount
6. loan_term
7. cibil_score
8. residential_assets_value
9. commercial_assets_value
10. luxury_assets_value
11. bank_asset_value

## Run Project Locally

Open VS Code Terminal in the project folder.

Install required libraries:

```bash
pip install -r requirements.txt