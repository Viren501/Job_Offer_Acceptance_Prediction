import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="HR Analytics & Prediction", page_icon="📊", layout="wide")

# --- DATA & MODEL LOADING ---
@st.cache_data
def load_data():
    """Loads the dataset for Dashboard and Analysis tabs."""
    # Attempt to load cleaned data for better visualization labels
    file_path = "../data/cleaned_hr_dataset.csv"
    # if not os.path.exists(file_path):
    #     file_path = "../data/hr_dataset.csv" # Fallback
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return pd.DataFrame() # Return empty if neither exists

@st.cache_resource
def load_models():
    """Loads the trained ML models and preprocessing objects."""
    try:
        model = joblib.load("../models/best_model.pkl")
        scaler = joblib.load("../models/scaler.pkl")
        columns = joblib.load("../models/feature_columns.pkl")
        return model, scaler, columns
    except Exception as e:
        return None, None, None

df = load_data()
model, scaler, feature_columns = load_models()

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135679.png", width=100)
st.sidebar.title("Navigation")
menu = ["Home", "Dashboard", "Predict", "Analysis", "Model Comparison", "About"]
choice = st.sidebar.radio("Go to:", menu)

st.sidebar.markdown("---")
st.sidebar.markdown("**Brainybeam Info-Tech PVT LTD**")

# --- 1. HOME TAB ---
if choice == "Home":
    st.title("🏢 HR Analytics & Joining Predictor")
    st.markdown("### Welcome to the HR Intelligence Portal")
    st.markdown("""
    This application is designed to empower the HR department with data-driven insights. 
    By leveraging historical hiring data and advanced machine learning models, this tool helps in predicting whether a prospective candidate will accept a job offer and join the organization.
    
    **Key Features:**
    * **Dashboard:** High-level overview of hiring metrics.
    * **Predict:** Real-time prediction engine for new candidates.
    * **Analysis:** Deep dive into the factors influencing candidate decisions.
    * **Model Comparison:** Transparency into the machine learning algorithms powering the predictions.
    """)
    st.info("👈 Please use the sidebar to navigate through the application.")

# --- 2. DASHBOARD TAB ---
elif choice == "Dashboard":
    st.title("📈 Executive Dashboard")
    
    if df.empty:
        st.error("Dataset not found. Please ensure 'cleaned_hr_dataset.csv' or 'hr_dataset.csv' is in the '../data/' folder.")
    else:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        total_candidates = len(df)
        
        # Assuming 'Status' is 1 for Joined, 0 for Not Joined, or string 'Joined'/'Not Joined'
        if 'Status' in df.columns:
            if df['Status'].dtype == 'O':
                joined_count = len(df[df['Status'].str.contains('Join', na=False)])
            else:
                joined_count = df['Status'].sum()
                
            join_rate = (joined_count / total_candidates) * 100
        else:
            joined_count = 0
            join_rate = 0

        col1.metric("Total Candidates", f"{total_candidates:,}")
        col2.metric("Total Joined", f"{joined_count:,}")
        col3.metric("Overall Joining Rate", f"{join_rate:.1f}%")
        col4.metric("Avg Notice Period", f"{df['Notice period'].mean():.0f} Days" if 'Notice period' in df.columns else "N/A")

        st.markdown("---")
        
        # High-level charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            if 'Status' in df.columns:
                status_counts = df['Status'].value_counts().reset_index()
                status_counts.columns = ['Status', 'Count']
                fig1 = px.pie(status_counts, values='Count', names='Status', title="Candidate Joining Distribution", 
                              color_discrete_sequence=['#0047AB', '#E9ECEF'], hole=0.4)
                st.plotly_chart(fig1, use_container_width=True)

        with col_chart2:
            if 'Candidate Source' in df.columns:
                source_counts = df['Candidate Source'].value_counts().reset_index()
                source_counts.columns = ['Source', 'Count']
                fig2 = px.bar(source_counts, x='Source', y='Count', title="Candidates by Source",
                              color_discrete_sequence=['#0047AB'])
                st.plotly_chart(fig2, use_container_width=True)

# --- 3. PREDICT TAB ---
elif choice == "Predict":
    st.title("🎯 Candidate Joining Predictor")
    st.markdown("Enter the candidate's details below to predict the likelihood of them joining.")
    
    if model is None:
        st.error("Model files not found. Please run the training notebook to generate the `.pkl` files.")
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
                location          = st.selectbox("Location", ["Noida", "Chennai", "Gurgaon", "Bangalore", "Hyderabad", "Kolkata", "Cochin", "Pune", "Ahmedabad", "Mumbai", "Others"])
                age               = st.number_input("Age", 18, 65, 28)
            
            submit_button = st.form_submit_button(label="Analyze & Predict")

        if submit_button:
            band_mapping = {"E0": 0.0, "E1": 1.0, "E2": 2.0, "E3": 3.0}
            
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
                "Offered band": band_mapping[offered_band],
                "Candidate Source": candidate_source,
                "LOB": lob,
                "Location": location,
            }

            input_df = pd.DataFrame([raw])
            input_df = pd.get_dummies(input_df, columns=["Candidate Source", "LOB", "Location"])
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)

            num_cols = ["Duration to accept offer", "Notice period", "Percent hike expected in CTC",
                        "Percent hike offered in CTC", "Percent difference CTC", "Rex in Yrs", "Age"]
            input_df[num_cols] = scaler.transform(input_df[num_cols])

            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]
            confidence = prob[pred] * 100

            st.markdown("### Prediction Result")
            if pred == 1:
                st.success(f"✅ The candidate is **Likely to JOIN** with a confidence of {confidence:.1f}%")
            else:
                st.error(f"❌ The candidate is **Likely to NOT JOIN** with a confidence of {confidence:.1f}%")
            
            st.progress(int(confidence))

# --- 4. ANALYSIS TAB ---
elif choice == "Analysis":
    st.title("🔍 Exploratory Data Analysis")
    
    if df.empty:
        st.warning("Data not available for analysis.")
    else:
        st.markdown("Explore how different features impact candidate decisions.")
        
        tab1, tab2 = st.tabs(["Categorical Variables", "Numerical Variables"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                if 'Location' in df.columns and 'Status' in df.columns:
                    fig = px.histogram(df, x="Location", color="Status", barmode="group", 
                                       title="Joining Status by Location", color_discrete_sequence=['#0047AB', '#ADB5BD'])
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'LOB' in df.columns and 'Status' in df.columns:
                    fig = px.histogram(df, x="LOB", color="Status", barmode="group", 
                                       title="Joining Status by Line of Business", color_discrete_sequence=['#0047AB', '#ADB5BD'])
                    st.plotly_chart(fig, use_container_width=True)
                    
        with tab2:
            col3, col4 = st.columns(2)
            with col3:
                if 'Notice period' in df.columns and 'Status' in df.columns:
                    fig = px.box(df, x="Status", y="Notice period", color="Status", 
                                 title="Notice Period vs. Joining Status", color_discrete_sequence=['#0047AB', '#ADB5BD'])
                    st.plotly_chart(fig, use_container_width=True)
            with col4:
                if 'Percent difference CTC' in df.columns and 'Status' in df.columns:
                    fig = px.box(df, x="Status", y="Percent difference CTC", color="Status", 
                                 title="CTC Difference vs. Joining Status", color_discrete_sequence=['#0047AB', '#ADB5BD'])
                    st.plotly_chart(fig, use_container_width=True)

# --- 5. MODEL COMPARISON TAB ---
elif choice == "Model Comparison":
    st.title("⚙️ Model Performance Comparison")
    st.markdown("This section details the performance of various machine learning algorithms tested during the training phase.")
    
    # Data compiled from 04model_training.ipynb outputs
    model_data = {
        "Model": ["Random Forest", "Decision Tree", "XGBoost", "KNN", "Gradient Boosting", "Logistic Regression", "AdaBoost", "Naive Bayes"],
        "Accuracy": [0.9466, 0.8992, 0.8573, 0.7727, 0.7312, 0.6778, 0.6465, 0.5854],
        "Precision": [0.9489, 0.9109, 0.8645, 0.7896, 0.7339, 0.6809, 0.6511, 0.7745],
        "Recall": [0.9466, 0.8992, 0.8573, 0.7727, 0.7312, 0.6778, 0.6465, 0.5854],
        "F1 Score": [0.9466, 0.8987, 0.8570, 0.7703, 0.7310, 0.6773, 0.6415, 0.5076]
    }
    
    results_df = pd.DataFrame(model_data)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Evaluation Metrics")
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'F1 Score'], color='#D4EDDA'), use_container_width=True)
        st.success("🏆 **Random Forest** was selected as the final production model due to the highest F1 Score (0.9466).")
        
    with col2:
        st.markdown("### F1 Score Visualization")
        fig = px.bar(results_df.sort_values('F1 Score', ascending=True), 
                     x='F1 Score', y='Model', orientation='h',
                     color='F1 Score', color_continuous_scale='Blues',
                     text_auto='.4f')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# --- 6. ABOUT TAB ---
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
    
    ### Organization
    Developed for internal use at **Brainybeam Info-Tech PVT LTD**.
    """)
    st.markdown("---")
    st.caption("Version 1.0.0 | Developed 2026")