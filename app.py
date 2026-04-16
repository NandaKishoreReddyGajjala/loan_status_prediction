import numpy as np
import joblib
import streamlit as st

# Load the trained SVM model
model = joblib.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Loan Status Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Loan Status Predictor")
st.markdown("Fill in the applicant details below to predict loan approval.")

st.divider()

# --- Personal Details ---
st.subheader("👤 Personal Details")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])

with col2:
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.divider()

# --- Financial Details ---
st.subheader("💰 Financial Details")
col3, col4 = st.columns(2)

with col3:
    applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands ₹)", min_value=0, value=150, step=10)
    credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Clear (1)" if x == 1 else "Not Clear (0)")

with col4:
    coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0, value=0, step=500)
    loan_amount_term = st.selectbox("Loan Amount Term (months)", [360, 120, 180, 240, 300, 480, 60, 36, 84])

st.divider()

# --- Predict ---
if st.button("🔍 Predict Loan Status", use_container_width=True, type="primary"):

    # Encode inputs (matching notebook preprocessing)
    gender_enc       = 1 if gender == "Male" else 0
    married_enc      = 1 if married == "Yes" else 0
    dependents_enc   = 4 if dependents == "3+" else int(dependents)
    education_enc    = 1 if education == "Graduate" else 0
    self_employed_enc = 1 if self_employed == "Yes" else 0
    property_area_enc = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

    input_data = np.array([[
        gender_enc,
        married_enc,
        dependents_enc,
        education_enc,
        self_employed_enc,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        property_area_enc
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ **Loan Approved!** The applicant is likely eligible for the loan.")
    else:
        st.error("❌ **Loan Not Approved.** The applicant does not meet the approval criteria.")