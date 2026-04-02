import streamlit as st
import pandas as pd
import joblib

st.title("Job Offer Acceptance Predictor")
st.subheader("Brainybeam Info-Tech PVT LTD")
st.markdown("Predict whether a candidate will **Join** or **Not Join** based on offer details.")

@st.cache_resource
def load_models():
    model   = joblib.load("../models/best_model.pkl")
    scaler  = joblib.load("../models/scaler.pkl")
    columns = joblib.load("../models/feature_columns.pkl")
    return model, scaler, columns

model, scaler, feature_columns = load_models()

# --- Input widgets ---
col1, col2, col3 = st.columns(3)

with col1:
    doj_extended      = st.selectbox("DOJ Extended", ["Yes", "No"])
    duration          = st.number_input("Duration to Accept Offer (days)", 1, 60, 10)
    notice_period     = st.number_input("Notice Period (days)", 0, 180, 30)
    offered_band      = st.selectbox("Offered Band", ["E0", "E1", "E2", "E3"])
    pct_hike_expected = st.number_input("% Hike Expected in CTC", -100.0, 500.0, 10.0)

with col2:
    pct_hike_offered  = st.number_input("% Hike Offered in CTC", -100.0, 500.0, 10.0)
    pct_diff_ctc      = st.number_input("% Difference CTC", -100.0, 500.0, 0.0)
    joining_bonus     = st.selectbox("Joining Bonus", ["Yes", "No"])
    relocate          = st.selectbox("Candidate Relocate Actual", ["Yes", "No"])
    gender            = st.selectbox("Gender", ["Male", "Female"])

with col3:
    candidate_source  = st.selectbox("Candidate Source", ["Agency", "Employee Referral", "Direct"])
    rex_in_yrs        = st.number_input("Experience (Rex in Yrs)", 0, 40, 3)
    lob               = st.selectbox("LOB", ["ERS", "INFRA", "Healthcare", "BFSI", "CSMP", "ETS", "AXON", "EAS"])
    location          = st.selectbox("Location", ["Noida", "Chennai", "Gurgaon", "Bangalore", "Hyderabad", "Kolkata", "Cochin", "Pune"])
    age               = st.number_input("Age", 18, 65, 28)

# --- Prediction ---
if st.button("Predict"):
    # Build raw input dict
    raw = {
        "DOJ Extended": 1 if doj_extended == "Yes" else 0,
        "Duration to accept offer": duration,
        "Notice period": notice_period,
        "Percent hike expected in CTC": pct_hike_expected,
        "Percent hike offered in CTC": pct_hike_offered,
        "Percent difference CTC": pct_diff_ctc,
        "Joining Bonus": 1 if joining_bonus == "Yes" else 0,
        "Candidate relocate actual": 1 if relocate == "Yes" else 0,
        "Gender": 1 if gender == "Male" else 0,
        "Rex in Yrs": rex_in_yrs,
        "Age": age,
        # OHE placeholders
        "Offered band": offered_band,
        "Candidate Source": candidate_source,
        "LOB": lob,
        "Location": location,
    }

    input_df = pd.DataFrame([raw])

    # One-hot encode categorical columns (match training)
    input_df = pd.get_dummies(input_df, columns=["Offered band", "Candidate Source", "LOB", "Location"], drop_first=True)

    # Align to training feature columns
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scale numerical features
    num_cols = ["Duration to accept offer", "Notice period", "Percent hike expected in CTC",
                "Percent hike offered in CTC", "Percent difference CTC", "Rex in Yrs", "Age"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]
    confidence = prob[pred] * 100

    st.markdown("---")
    if pred == 1:
        st.success(f"✅ Likely to JOIN ({confidence:.1f}% confidence)")
    else:
        st.error(f"❌ Likely to NOT JOIN ({confidence:.1f}% confidence)")

    st.progress(int(confidence))
