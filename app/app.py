import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import io
import json
import sqlite3
import tempfile
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Job Offer Acceptance Prediction", page_icon="📊", layout="wide")

# --- DATA & MODEL LOADING ---
@st.cache_data
def load_data():
    """Loads the dataset for Dashboard and Analysis tabs."""
    file_path = "../data/cleaned_hr_dataset.csv"
    if not os.path.exists(file_path):
        file_path = "../data/hr_dataset.csv"
    try:
        df = pd.read_csv(file_path)
        if 'Status' in df.columns:
            if pd.api.types.is_numeric_dtype(df['Status']):
                df['Status_Label'] = df['Status'].map({1: 'Joined', 0: 'Not Joined'})
            else:
                df['Status_Label'] = df['Status'].astype(str).str.strip().str.title()
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_resource
def load_models():
    """Loads the trained ML model, scaler, feature columns, and ALL encoders."""
    try:BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
model          = joblib.load(os.path.join(BASE_DIR, "../models/best_model.pkl"))
scaler         = joblib.load(os.path.join(BASE_DIR, "../models/scaler.pkl"))
feature_cols   = joblib.load(os.path.join(BASE_DIR, "../models/feature_columns.pkl"))
le_encoders    = joblib.load(os.path.join(BASE_DIR, "../models/le_encoders.pkl"))
oe_band        = joblib.load(os.path.join(BASE_DIR, "../models/oe_offered_band.pkl"))
ohe            = joblib.load(os.path.join(BASE_DIR, "../models/ohe_encoder.pkl"))
        return model, scaler, feature_cols, le_encoders, oe_band, ohe
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None, None, None, None

df = load_data()
model, scaler, feature_columns, le_encoders, oe_band, ohe = load_models()

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135679.png", width=100)
st.sidebar.title("Navigation")
menu = ["Home", "Dashboard", "Predict", "Bulk Scanner", "Analysis", "Model Comparison", "About"]
choice = st.sidebar.radio("🧭", menu)
st.sidebar.markdown("---")
st.sidebar.markdown("")

# ==============================================================================
# BULK SCANNER HELPERS
# ==============================================================================

# Exact raw column names the model was trained on
BULK_FEATURE_COLS = [
    "DOJ Extended",
    "Duration to accept offer",
    "Notice period",
    "Offered band",
    "Percent hike expected in CTC",
    "Percent hike offered in CTC",
    "Percent difference CTC",
    "Joining Bonus",
    "Candidate relocate actual",
    "Gender",
    "Rex in Yrs",
    "Age",
    "Candidate Source",
    "LOB",
    "Location",
]

SAMPLE_ROWS = [
    {
        "DOJ Extended": "No",
        "Duration to accept offer": 7,
        "Notice period": 30,
        "Offered band": "E1",
        "Percent hike expected in CTC": 20.0,
        "Percent hike offered in CTC": 25.0,
        "Percent difference CTC": 5.0,
        "Joining Bonus": "No",
        "Candidate relocate actual": "Yes",
        "Gender": "Male",
        "Rex in Yrs": 5,
        "Age": 28,
        "Candidate Source": "Agency",
        "LOB": "ERS",
        "Location": "Bangalore",
    },
    {
        "DOJ Extended": "Yes",
        "Duration to accept offer": 14,
        "Notice period": 60,
        "Offered band": "E2",
        "Percent hike expected in CTC": 30.0,
        "Percent hike offered in CTC": 28.0,
        "Percent difference CTC": -2.0,
        "Joining Bonus": "Yes",
        "Candidate relocate actual": "No",
        "Gender": "Female",
        "Rex in Yrs": 9,
        "Age": 34,
        "Candidate Source": "Employee Referral",
        "LOB": "BFSI",
        "Location": "Mumbai",
    },
    {
        "DOJ Extended": "No",
        "Duration to accept offer": 3,
        "Notice period": 15,
        "Offered band": "E0",
        "Percent hike expected in CTC": 40.0,
        "Percent hike offered in CTC": 35.0,
        "Percent difference CTC": -5.0,
        "Joining Bonus": "No",
        "Candidate relocate actual": "Yes",
        "Gender": "Male",
        "Rex in Yrs": 2,
        "Age": 24,
        "Candidate Source": "Direct",
        "LOB": "Healthcare",
        "Location": "Hyderabad",
    },
    {
        "DOJ Extended": "No",
        "Duration to accept offer": 10,
        "Notice period": 45,
        "Offered band": "E3",
        "Percent hike expected in CTC": 15.0,
        "Percent hike offered in CTC": 22.0,
        "Percent difference CTC": 7.0,
        "Joining Bonus": "Yes",
        "Candidate relocate actual": "No",
        "Gender": "Female",
        "Rex in Yrs": 12,
        "Age": 38,
        "Candidate Source": "Agency",
        "LOB": "INFRA",
        "Location": "Noida",
    },
    {
        "DOJ Extended": "Yes",
        "Duration to accept offer": 20,
        "Notice period": 90,
        "Offered band": "E2",
        "Percent hike expected in CTC": 50.0,
        "Percent hike offered in CTC": 30.0,
        "Percent difference CTC": -20.0,
        "Joining Bonus": "No",
        "Candidate relocate actual": "No",
        "Gender": "Male",
        "Rex in Yrs": 7,
        "Age": 31,
        "Candidate Source": "Direct",
        "LOB": "CSMP",
        "Location": "Chennai",
    },
]

NUM_COLS_SCALE = [
    "Duration to accept offer",
    "Notice period",
    "Percent hike expected in CTC",
    "Percent hike offered in CTC",
    "Percent difference CTC",
    "Rex in Yrs",
    "Age",
]


def encode_bulk_row(row):
    """Encode a single raw row (as dict/Series) the same way the single-predict tab does."""
    gender_encoded   = 1 if str(row.get("Gender", "")).strip().title() == "Male" else 0

    doj_val = str(row.get("DOJ Extended", "No")).strip().title()
    bonus_val = str(row.get("Joining Bonus", "No")).strip().title()
    relocate_val = str(row.get("Candidate relocate actual", "No")).strip().title()

    try:
        doj_encoded      = le_encoders['DOJ Extended'].transform([doj_val])[0]
    except Exception:
        doj_encoded      = 0
    try:
        bonus_encoded    = le_encoders['Joining Bonus'].transform([bonus_val])[0]
    except Exception:
        bonus_encoded    = 0
    try:
        relocate_encoded = le_encoders['Candidate relocate actual'].transform([relocate_val])[0]
    except Exception:
        relocate_encoded = 0

    band_val = str(row.get("Offered band", "E0")).strip().upper()
    try:
        band_encoded = oe_band.transform([[band_val]])[0][0]
    except Exception:
        band_encoded = 0

    numeric_row = {
        "DOJ Extended":                 doj_encoded,
        "Duration to accept offer":     float(row.get("Duration to accept offer", 10)),
        "Notice period":                float(row.get("Notice period", 30)),
        "Offered band":                 band_encoded,
        "Percent hike expected in CTC": float(row.get("Percent hike expected in CTC", 10)),
        "Percent hike offered in CTC":  float(row.get("Percent hike offered in CTC", 10)),
        "Percent difference CTC":       float(row.get("Percent difference CTC", 0)),
        "Joining Bonus":                bonus_encoded,
        "Candidate relocate actual":    relocate_encoded,
        "Gender":                       gender_encoded,
        "Rex in Yrs":                   float(row.get("Rex in Yrs", 3)),
        "Age":                          float(row.get("Age", 28)),
    }
    return numeric_row


def predict_bulk(upload_df: pd.DataFrame):
    """Run the full encoding + scaling + model pipeline on a DataFrame of raw rows."""
    numeric_rows = []
    nominal_rows = []

    for _, row in upload_df.iterrows():
        numeric_rows.append(encode_bulk_row(row))
        nominal_rows.append({
            "Candidate Source": str(row.get("Candidate Source", "Agency")).strip(),
            "LOB":              str(row.get("LOB", "ERS")).strip(),
            "Location":         str(row.get("Location", "Noida")).strip(),
        })

    input_num_df = pd.DataFrame(numeric_rows)
    nominal_df   = pd.DataFrame(nominal_rows, columns=["Candidate Source", "LOB", "Location"])

    try:
        ohe_values = ohe.transform(nominal_df)
        ohe_cols   = ohe.get_feature_names_out(["Candidate Source", "LOB", "Location"])
        ohe_df     = pd.DataFrame(ohe_values, columns=ohe_cols)
    except Exception as e:
        st.error(f"OneHot encoding failed: {e}. Check that Candidate Source / LOB / Location values match training data.")
        return None

    combined = pd.concat([input_num_df.reset_index(drop=True),
                          ohe_df.reset_index(drop=True)], axis=1)
    combined = combined.reindex(columns=feature_columns, fill_value=0)
    combined[NUM_COLS_SCALE] = scaler.transform(combined[NUM_COLS_SCALE])

    preds  = model.predict(combined)
    probas = model.predict_proba(combined)

    result_df = upload_df.copy().reset_index(drop=True)
    result_df["Prediction"]  = ["✅ Joined" if p == 1 else "❌ Not Joined" for p in preds]
    result_df["Confidence %"] = [f"{probas[i][p]*100:.1f}%" for i, p in enumerate(preds)]
    result_df["Confidence_raw"] = [probas[i][p] for i, p in enumerate(preds)]

    return result_df


def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        elif name.endswith(".json"):
            content = json.load(uploaded_file)
            return pd.DataFrame(content if isinstance(content, list) else [content])
        elif name.endswith((".db", ".sqlite", ".sql")):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            conn   = sqlite3.connect(tmp_path)
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            conn.close()
            if tables.empty:
                st.error("No tables found in the SQLite database.")
                os.unlink(tmp_path)
                return None
            table_name = st.selectbox("Select table to scan", tables["name"].tolist())
            conn2 = sqlite3.connect(tmp_path)
            result = pd.read_sql(f'SELECT * FROM "{table_name}"', conn2)
            conn2.close()
            os.unlink(tmp_path)
            return result
        else:
            st.error(f"Unsupported file type: **{uploaded_file.name}**")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def df_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")


def df_to_excel(dataframe):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Scan Results")
    return buf.getvalue()


def df_to_json(dataframe):
    return dataframe.to_json(orient="records", indent=2).encode("utf-8")


def get_sample_df():
    return pd.DataFrame(SAMPLE_ROWS)[BULK_FEATURE_COLS]


# ==============================================================================
# 1. HOME TAB
# ==============================================================================
if choice == "Home":
    st.title("🏢 Job Offer Acceptance Prediction")
    st.markdown("### Welcome to the HR Intelligence Portal")
    st.markdown("""
    This application is designed to empower the HR department with data-driven insights. 
    By leveraging historical hiring data and advanced machine learning models, this tool helps in predicting whether a prospective candidate will accept a job offer and join the organization.
    
    **Key Features:**
    * **Dashboard:** High-level overview of hiring metrics.
    * **Predict:** Real-time prediction engine for new candidates.
    * **Bulk Scanner:** Upload CSV/Excel/JSON/SQLite files to scan multiple candidates at once and download results.
    * **Analysis:** Comprehensive Exploratory Data Analysis (EDA) of all hiring factors.
    * **Model Comparison:** Transparency into the machine learning algorithms powering the predictions.
    """)
    st.info("👈 Please use the sidebar to navigate through the application.")

# ==============================================================================
# 2. DASHBOARD TAB
# ==============================================================================
elif choice == "Dashboard":
    st.title("📈 Executive Dashboard")

    if df.empty:
        st.error("Dataset not found. Please ensure 'cleaned_hr_dataset.csv' or 'hr_dataset.csv' is in the '../data/' folder.")
    else:
        col1, col2, col3 = st.columns(3)
        total_candidates = len(df)

        if 'Status' in df.columns:
            if pd.api.types.is_numeric_dtype(df['Status']):
                joined_count = (df['Status'] == 1).sum()
            else:
                joined_count = df['Status'].astype(str).str.strip().str.lower().isin(['joined', '1', 'yes']).sum()
            join_rate = (joined_count / total_candidates) * 100
        else:
            joined_count = 0
            join_rate = 0

        hike_gap_text = "N/A"
        if 'Percent hike expected in CTC' in df.columns and 'Percent hike offered in CTC' in df.columns:
            hike_gap = df['Percent hike offered in CTC'].mean() - df['Percent hike expected in CTC'].mean()
            hike_gap_text = f"Avg Hike Gap: {hike_gap:+.1f}%"

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Candidates", f"{total_candidates:,}")
        col2.metric("Total Joined", f"{joined_count:,}")
        col3.metric(label="Overall Joining Rate", value=f"{join_rate:.1f}%",
                    delta=hike_gap_text, delta_color="off")
        st.markdown("---")

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            if 'Status_Label' in df.columns:
                status_counts = df['Status_Label'].value_counts().reset_index()
                status_counts.columns = ['Status', 'Count']
                fig1 = px.pie(status_counts, values='Count', names='Status',
                              title="Candidate Joining Distribution",
                              color_discrete_sequence=['#0047AB', '#E9ECEF'], hole=0.4)
                st.plotly_chart(fig1, use_container_width=True)
        with col_chart2:
            if 'Candidate Source' in df.columns:
                source_counts = df['Candidate Source'].value_counts().reset_index()
                source_counts.columns = ['Source', 'Count']
                fig2 = px.bar(source_counts, x='Source', y='Count', title="Candidates by Source",
                              color_discrete_sequence=['#0047AB'])
                st.plotly_chart(fig2, use_container_width=True)

# ==============================================================================
# 3. PREDICT TAB
# ==============================================================================
elif choice == "Predict":
    st.title("🎯 Candidate Joining Predictor")
    st.markdown("Enter the candidate's details below to predict the likelihood of them joining.")

    if model is None:
        st.error("Model files not found. Please run the training notebook and `save_encoders.py` to generate all `.pkl` files.")
    else:
        with st.form("prediction_form"):
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
                lob               = st.selectbox("LOB", ["ERS", "INFRA", "Healthcare", "BFSI", "CSMP", "ETS", "AXON", "EAS", "MMS"])
                location          = st.selectbox("Location", ["Noida", "Chennai", "Gurgaon", "Bangalore",
                                                               "Hyderabad", "Kolkata", "Cochin", "Pune",
                                                               "Ahmedabad", "Mumbai", "Others"])
                age               = st.number_input("Age", 18, 65, 28)

            submit_button = st.form_submit_button(label="Analyze & Predict")

        if submit_button:
            gender_encoded   = 1 if gender == "Male" else 0
            doj_encoded      = le_encoders['DOJ Extended'].transform([doj_extended])[0]
            bonus_encoded    = le_encoders['Joining Bonus'].transform([joining_bonus])[0]
            relocate_encoded = le_encoders['Candidate relocate actual'].transform([relocate])[0]
            band_encoded     = oe_band.transform([[offered_band]])[0][0]

            numeric_row = {
                "DOJ Extended":                doj_encoded,
                "Duration to accept offer":    duration,
                "Notice period":               notice_period,
                "Offered band":                band_encoded,
                "Percent hike expected in CTC": pct_hike_expected,
                "Percent hike offered in CTC":  pct_hike_offered,
                "Percent difference CTC":       pct_diff_ctc,
                "Joining Bonus":               bonus_encoded,
                "Candidate relocate actual":   relocate_encoded,
                "Gender":                      gender_encoded,
                "Rex in Yrs":                  rex_in_yrs,
                "Age":                         age,
            }
            input_df = pd.DataFrame([numeric_row])

            nominal_input = pd.DataFrame([[candidate_source, lob, location]],
                                         columns=['Candidate Source', 'LOB', 'Location'])
            ohe_values  = ohe.transform(nominal_input)
            ohe_cols    = ohe.get_feature_names_out(['Candidate Source', 'LOB', 'Location'])
            ohe_df      = pd.DataFrame(ohe_values, columns=ohe_cols)

            input_df = pd.concat([input_df.reset_index(drop=True),
                                  ohe_df.reset_index(drop=True)], axis=1)
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)

            num_cols = ["Duration to accept offer", "Notice period",
                        "Percent hike expected in CTC", "Percent hike offered in CTC",
                        "Percent difference CTC", "Rex in Yrs", "Age"]
            input_df[num_cols] = scaler.transform(input_df[num_cols])

            pred       = model.predict(input_df)[0]
            prob       = model.predict_proba(input_df)[0]
            confidence = prob[pred] * 100

            st.markdown("### Prediction Result")
            if pred == 1:
                st.success(f"✅ The candidate is **Likely to JOIN** with a confidence of {confidence:.1f}%")
            else:
                st.error(f"❌ The candidate is **Likely to NOT JOIN** with a confidence of {confidence:.1f}%")

            st.progress(int(confidence))

            with st.expander("🔍 Show encoded input (for debugging)"):
                st.dataframe(input_df)

# ==============================================================================
# 4. BULK SCANNER TAB
# ==============================================================================
elif choice == "Bulk Scanner":
    st.title("Bulk Scanner")
    st.markdown(
        "Upload a file containing **multiple candidates** and get batch joining predictions instantly. "
        "Supported formats: **CSV · Excel (.xlsx) · JSON · SQLite (.db / .sqlite)**"
    )

    if model is None:
        st.error("❌ Model files not found. Cannot run predictions. Please ensure all `.pkl` files are present.")
        st.stop()

    # ── Sample file download ─────────────────────────────────────────────────
    with st.expander("📥 Download Sample Input File", expanded=True):
        st.markdown(
            "**New here?** Download a sample file to see exactly which columns are required "
            "and what values are accepted. Fill in your data using the same format and upload it below."
        )

        sample_df = get_sample_df()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                label="⬇️ Sample CSV",
                data=df_to_csv(sample_df),
                file_name="sample_candidates.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                label="⬇️ Sample Excel",
                data=df_to_excel(sample_df),
                file_name="sample_candidates.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with c3:
            st.download_button(
                label="⬇️ Sample JSON",
                data=df_to_json(sample_df),
                file_name="sample_candidates.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("**Required columns (15 total):**")
        st.code(", ".join(BULK_FEATURE_COLS), language=None)

        st.markdown("""
        **Allowed values:**
        | Column | Accepted values |
        |---|---|
        | DOJ Extended | `Yes` / `No` |
        | Offered band | `E0` / `E1` / `E2` / `E3` |
        | Joining Bonus | `Yes` / `No` |
        | Candidate relocate actual | `Yes` / `No` |
        | Gender | `Male` / `Female` |
        | Candidate Source | `Agency` / `Employee Referral` / `Direct` |
        | LOB | `ERS` / `INFRA` / `Healthcare` / `BFSI` / `CSMP` / `ETS` / `AXON` / `EAS` / `MMS` |
        | Location | `Noida` / `Chennai` / `Gurgaon` / `Bangalore` / `Hyderabad` / `Kolkata` / `Cochin` / `Pune` / `Ahmedabad` / `Mumbai` / `Others` |
        """)

    st.divider()

    # ── File upload ──────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📁 Upload your candidate file",
        type=["csv", "xlsx", "xls", "json", "db", "sqlite", "sql"],
        help="Upload a file with candidate records. Each row = one candidate.",
    )

    if uploaded_file is None:
        st.info("👆 Upload a file above to start bulk scanning.")
        st.stop()

    # ── Read file ────────────────────────────────────────────────────────────
    with st.spinner("Reading file…"):
        upload_df = read_uploaded_file(uploaded_file)

    if upload_df is None or upload_df.empty:
        st.error("Could not read data from the file, or the file is empty.")
        st.stop()

    st.success(f"✅ File loaded — **{len(upload_df):,} rows** × **{len(upload_df.columns)} columns**")

    # ── Column check ─────────────────────────────────────────────────────────
    missing_cols = [c for c in BULK_FEATURE_COLS if c not in upload_df.columns]
    if missing_cols:
        st.error(f"❌ Missing required column(s): `{'`, `'.join(missing_cols)}`")
        st.markdown("Please fix the file or download a sample file above.")
        with st.expander("Preview your uploaded file"):
            st.dataframe(upload_df.head(10), use_container_width=True)
        st.stop()

    extra_cols = [c for c in upload_df.columns if c not in BULK_FEATURE_COLS]
    if extra_cols:
        st.warning(f"⚠️ Extra columns found (will be included in output): `{'`, `'.join(extra_cols)}`")

    with st.expander("👀 Preview uploaded data (first 5 rows)"):
        st.dataframe(upload_df.head(5), use_container_width=True)

    # ── Run Scan button ──────────────────────────────────────────────────────
    if st.button("🚀 Run Bulk Scan", type="primary", use_container_width=True):

        progress_bar = st.progress(0, text="Preparing data…")
        with st.spinner("Running predictions on all candidates…"):
            try:
                progress_bar.progress(30, text="Encoding features…")
                result_df = predict_bulk(upload_df)
                progress_bar.progress(80, text="Generating results…")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        if result_df is None:
            st.stop()

        progress_bar.progress(100, text="Done!")

        # ── Summary metrics ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Scan Summary")

        joined_count  = (result_df["Prediction"] == "✅ Joined").sum()
        not_join_count = (result_df["Prediction"] == "❌ Not Joined").sum()
        avg_conf      = result_df["Confidence_raw"].mean() * 100
        join_rate_bulk = joined_count / len(result_df) if len(result_df) else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📋 Total Candidates", f"{len(result_df):,}")
        m2.metric("✅ Likely to Join",    f"{joined_count:,}")
        m3.metric("❌ Likely to Drop",    f"{not_join_count:,}")
        m4.metric("🎯 Avg Confidence",    f"{avg_conf:.1f}%")

        st.markdown(f"**Predicted Joining Rate**")
        st.progress(join_rate_bulk, text=f"{join_rate_bulk:.1%} candidates predicted to join")

        # ── Charts ───────────────────────────────────────────────────────────
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            pred_counts = result_df["Prediction"].value_counts().reset_index()
            pred_counts.columns = ["Prediction", "Count"]
            fig_pie = px.pie(
                pred_counts, values="Count", names="Prediction",
                title="Joining Prediction Distribution",
                color="Prediction",
                color_discrete_map={"✅ Joined": "#0047AB", "❌ Not Joined": "#ADB5BD"},
                hole=0.4,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            fig_hist = px.histogram(
                result_df, x="Confidence_raw", color="Prediction",
                nbins=20, title="Confidence Score Distribution",
                labels={"Confidence_raw": "Confidence"},
                color_discrete_map={"✅ Joined": "#0047AB", "❌ Not Joined": "#ADB5BD"},
            )
            fig_hist.update_layout(xaxis_tickformat=".0%")
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Results table ────────────────────────────────────────────────────
        st.subheader("🔍 Candidate-Level Results")

        display_df = result_df.drop(columns=["Confidence_raw"])

        def highlight_prediction(val):
            if "Joined" in str(val) and "Not" not in str(val):
                return "background-color: #d4edda; color: #155724; font-weight: bold"
            elif "Not Joined" in str(val):
                return "background-color: #f8d7da; color: #721c24; font-weight: bold"
            return ""

        styled = display_df.style.applymap(highlight_prediction, subset=["Prediction"])
        st.dataframe(styled, use_container_width=True, height=420)

        # ── Download results ─────────────────────────────────────────────────
        st.subheader("⬇️ Download Scanned Results")
        st.markdown("Download the results file (includes all original columns + Prediction + Confidence %)")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            st.download_button(
                label="⬇️ Download as CSV",
                data=df_to_csv(display_df),
                file_name=f"bulk_scan_results_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                label="⬇️ Download as Excel",
                data=df_to_excel(display_df),
                file_name=f"bulk_scan_results_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with dl3:
            st.download_button(
                label="⬇️ Download as JSON",
                data=df_to_json(display_df),
                file_name=f"bulk_scan_results_{ts}.json",
                mime="application/json",
                use_container_width=True,
            )

# ==============================================================================
# 5. ANALYSIS TAB
# ==============================================================================
elif choice == "Analysis":
    st.title("🔍 Exploratory Data Analysis (EDA)")

    if df.empty or 'Status_Label' not in df.columns:
        st.warning("Data not available or missing 'Status' column for analysis.")
    else:
        st.markdown("Explore how different features impact candidate decisions across the entire historical dataset.")
        color_map = {'Joined': '#0047AB', 'Not Joined': '#ADB5BD'}

        tab1, tab2, tab3 = st.tabs(["👥 Demographics", "🏢 Role & Location", "💰 Offer Financials"])

        with tab1:
            st.markdown("#### Candidate Demographics & Experience")
            col1, col2 = st.columns(2)
            with col1:
                if 'Gender' in df.columns:
                    fig = px.histogram(df, x="Gender", color="Status_Label", barmode="group",
                                       title="Joining Status by Gender", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)
                if 'Rex in Yrs' in df.columns:
                    fig = px.box(df, x="Status_Label", y="Rex in Yrs", color="Status_Label",
                                 title="Relevant Experience (Years) vs Joining", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'Age' in df.columns:
                    fig = px.box(df, x="Status_Label", y="Age", color="Status_Label",
                                 title="Age Distribution vs Joining", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)
                if 'Candidate relocate actual' in df.columns:
                    df['Relocate'] = df['Candidate relocate actual'].apply(
                        lambda x: 'Yes' if str(x) == '1' else ('No' if str(x) == '0' else x))
                    fig = px.histogram(df, x="Relocate", color="Status_Label", barmode="group",
                                       title="Willingness to Relocate vs Joining", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("#### Job Position & Geography")
            col3, col4 = st.columns(2)
            with col3:
                if 'Location' in df.columns:
                    fig = px.histogram(df, x="Location", color="Status_Label", barmode="group",
                                       title="Joining Status by Location", color_discrete_map=color_map)
                    fig.update_layout(xaxis={'categoryorder': 'total descending'})
                    st.plotly_chart(fig, use_container_width=True)
                if 'Offered band' in df.columns:
                    fig = px.histogram(df, x="Offered band", color="Status_Label", barmode="group",
                                       title="Joining Status by Offered Band", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)
            with col4:
                if 'LOB' in df.columns:
                    fig = px.histogram(df, x="LOB", color="Status_Label", barmode="group",
                                       title="Joining Status by Line of Business (LOB)", color_discrete_map=color_map)
                    fig.update_layout(xaxis={'categoryorder': 'total descending'})
                    st.plotly_chart(fig, use_container_width=True)
                if 'Candidate Source' in df.columns:
                    fig = px.histogram(df, x="Candidate Source", color="Status_Label", barmode="group",
                                       title="Joining Status by Candidate Source", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown("#### Compensation, Hikes, & Timelines")
            col5, col6 = st.columns(2)
            with col5:
                if 'Percent difference CTC' in df.columns:
                    fig = px.box(df, x="Status_Label", y="Percent difference CTC", color="Status_Label",
                                 title="% Difference in CTC vs. Joining Status", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)
                if 'Percent hike offered in CTC' in df.columns:
                    fig = px.box(df, x="Status_Label", y="Percent hike offered in CTC", color="Status_Label",
                                 title="% Hike Offered vs. Joining Status", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)
                if 'Notice period' in df.columns:
                    fig = px.box(df, x="Status_Label", y="Notice period", color="Status_Label",
                                 title="Notice Period vs. Joining Status", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)
            with col6:
                if 'Percent hike expected in CTC' in df.columns:
                    fig = px.box(df, x="Status_Label", y="Percent hike expected in CTC", color="Status_Label",
                                 title="% Hike Expected vs. Joining Status", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)
                if 'Duration to accept offer' in df.columns:
                    fig = px.box(df, x="Status_Label", y="Duration to accept offer", color="Status_Label",
                                 title="Duration to Accept Offer (Days)", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)
                if 'Joining Bonus' in df.columns:
                    df['Bonus'] = df['Joining Bonus'].apply(
                        lambda x: 'Yes' if str(x) == '1' else ('No' if str(x) == '0' else x))
                    fig = px.histogram(df, x="Bonus", color="Status_Label", barmode="group",
                                       title="Joining Bonus Offered vs Joining", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 6. MODEL COMPARISON TAB
# ==============================================================================
elif choice == "Model Comparison":
    st.title("⚙️ Model Performance Comparison")
    st.markdown("This section details the performance of various machine learning algorithms tested during the training phase.")

    model_data = {
        "Model": ["Random Forest", "Decision Tree", "XGBoost", "KNN", "Gradient Boosting",
                  "Logistic Regression", "AdaBoost", "Naive Bayes"],
        "Accuracy":  [0.9466, 0.8992, 0.8573, 0.7727, 0.7312, 0.6778, 0.6465, 0.5854],
        "Precision": [0.9489, 0.9109, 0.8645, 0.7896, 0.7339, 0.6809, 0.6511, 0.7745],
        "Recall":    [0.9466, 0.8992, 0.8573, 0.7727, 0.7312, 0.6778, 0.6465, 0.5854],
        "F1 Score":  [0.9466, 0.8987, 0.8570, 0.7703, 0.7310, 0.6773, 0.6415, 0.5076],
    }
    results_df = pd.DataFrame(model_data)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Evaluation Metrics")
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'F1 Score'],
                                                     color='#D4EDDA'), use_container_width=True)
        st.success("🏆 **Random Forest** was selected as the final production model due to the highest F1 Score (0.9466).")
    with col2:
        st.markdown("### F1 Score Visualization")
        fig = px.bar(results_df.sort_values('F1 Score', ascending=True),
                     x='F1 Score', y='Model', orientation='h',
                     color='F1 Score', color_continuous_scale='Blues', text_auto='.4f')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 7. ABOUT TAB
# ==============================================================================
elif choice == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### Project Overview
    The **Job Offer Acceptance Predictor** is an analytical tool engineered to mitigate hiring friction. Offer dropouts cost organizations significant time and resources. This model leverages historical data points such as compensation differences, notice periods, and demographic factors to evaluate dropout risks before they happen.
    
    ### Tech Stack
    * **Frontend & Routing:** Streamlit
    * **Data Processing:** Pandas, Scikit-Learn
    * **Machine Learning Model:** Random Forest Classifier
    * **Visualizations:** Plotly Express
    """)
    st.markdown("---")
    st.markdown("## **Viren Vairagi**")
    st.caption("Internship Project")
