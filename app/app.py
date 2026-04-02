import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Job Offer Acceptance Prediction", page_icon="📊", layout="wide")

# --- DATA & MODEL LOADING ---
@st.cache_data
def load_data():
    """Loads the dataset for Dashboard and Analysis tabs."""
    # Get the directory where app.py is currently running
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the dynamic path to the data folder
    file_path = os.path.join(current_dir, "..", "data", "cleaned_hr_dataset.csv")
    
    if not os.path.exists(file_path):
        file_path = os.path.join(current_dir, "..", "data", "hr_dataset.csv") # Fallback
    
    try:
        df = pd.read_csv(file_path)
        # Create a display-friendly status column for plots
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
    """Loads the trained ML models and preprocessing objects."""
    # Get the directory where app.py is currently running
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        model = joblib.load(os.path.join(current_dir, "..", "models", "best_model.pkl"))
        scaler = joblib.load(os.path.join(current_dir, "..", "models", "scaler.pkl"))
        columns = joblib.load(os.path.join(current_dir, "..", "models", "feature_columns.pkl"))
        return model, scaler, columns
    except Exception as e:
        return None, None, None

df = load_data()
model, scaler, feature_columns = load_models()

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135679.png", width=100)
st.sidebar.title("Navigation")
menu = ["Home", "Dashboard", "Predict", "Analysis", "Model Comparison", "About"]
choice = st.sidebar.radio("🧭", menu)

st.sidebar.markdown("---")
st.sidebar.markdown("")

# --- 1. HOME TAB ---
if choice == "Home":
    st.title("🏢 Job Offer Acceptance Prediction")
    st.markdown("### Welcome to the HR Intelligence Portal")
    st.markdown("""
    This application is designed to empower the HR department with data-driven insights. 
    By leveraging historical hiring data and advanced machine learning models, this tool helps in predicting whether a prospective candidate will accept a job offer and join the organization.
    
    **Key Features:**
    * **Dashboard:** High-level overview of hiring metrics.
    * **Predict:** Real-time prediction engine for new candidates.
    * **Analysis:** Comprehensive Exploratory Data Analysis (EDA) of all hiring factors.
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
        col1, col2, col3 = st.columns(3)
        total_candidates = len(df)
        
        # Robust Status Counting
        if 'Status' in df.columns:
            if pd.api.types.is_numeric_dtype(df['Status']):
                joined_count = (df['Status'] == 1).sum()
            else:
                joined_count = df['Status'].astype(str).str.strip().str.lower().isin(['joined', '1', 'yes']).sum()
            join_rate = (joined_count / total_candidates) * 100
        else:
            joined_count = 0
            join_rate = 0

        # Calculate the Average Hike Gap (Offered - Expected)
        hike_gap_text = "N/A"
        if 'Percent hike expected in CTC' in df.columns and 'Percent hike offered in CTC' in df.columns:
            avg_expected = df['Percent hike expected in CTC'].mean()
            avg_offered = df['Percent hike offered in CTC'].mean()
            hike_gap = avg_offered - avg_expected
            
            # Format to show a + or - sign clearly
            hike_gap_text = f"Avg Hike Gap: {hike_gap:+.1f}%"

        # Update the layout to use 3 wider columns instead of 4
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Total Candidates", f"{total_candidates:,}")
        col2.metric("Total Joined", f"{joined_count:,}")
        
        # Combined KPI: Joining Rate with Hike Gap as the context/delta
        col3.metric(
            label="Overall Joining Rate", 
            value=f"{join_rate:.1f}%",
            delta=hike_gap_text,
            delta_color="off" # Prevents it from coloring green/red automatically, keeping it neutral
        )
        st.markdown("---")
        
        # High-level charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            if 'Status_Label' in df.columns:
                status_counts = df['Status_Label'].value_counts().reset_index()
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

# --- 4. ANALYSIS TAB (FULL EDA) ---
elif choice == "Analysis":
    st.title("🔍 Exploratory Data Analysis (EDA)")
    
    if df.empty or 'Status_Label' not in df.columns:
        st.warning("Data not available or missing 'Status' column for analysis.")
    else:
        st.markdown("Explore how different features impact candidate decisions across the entire historical dataset.")
        
        # Define standard colors for Joined vs Not Joined
        color_map = {'Joined': '#0047AB', 'Not Joined': '#ADB5BD'}
        
        tab1, tab2, tab3 = st.tabs(["👥 Demographics", "🏢 Role & Location", "💰 Offer Financials"])
        
        # --- SUB-TAB 1: DEMOGRAPHICS ---
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
                    # Convert 1/0 to Yes/No if needed
                    df['Relocate'] = df['Candidate relocate actual'].apply(lambda x: 'Yes' if str(x)=='1' else ('No' if str(x)=='0' else x))
                    fig = px.histogram(df, x="Relocate", color="Status_Label", barmode="group", 
                                       title="Willingness to Relocate vs Joining", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)

        # --- SUB-TAB 2: ROLE & LOCATION ---
        with tab2:
            st.markdown("#### Job Position & Geography")
            col3, col4 = st.columns(2)
            with col3:
                if 'Location' in df.columns:
                    fig = px.histogram(df, x="Location", color="Status_Label", barmode="group", 
                                       title="Joining Status by Location", color_discrete_map=color_map)
                    fig.update_layout(xaxis={'categoryorder':'total descending'})
                    st.plotly_chart(fig, use_container_width=True)
                if 'Offered band' in df.columns:
                    fig = px.histogram(df, x="Offered band", color="Status_Label", barmode="group", 
                                       title="Joining Status by Offered Band", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)
            with col4:
                if 'LOB' in df.columns:
                    fig = px.histogram(df, x="LOB", color="Status_Label", barmode="group", 
                                       title="Joining Status by Line of Business (LOB)", color_discrete_map=color_map)
                    fig.update_layout(xaxis={'categoryorder':'total descending'})
                    st.plotly_chart(fig, use_container_width=True)
                if 'Candidate Source' in df.columns:
                    fig = px.histogram(df, x="Candidate Source", color="Status_Label", barmode="group", 
                                       title="Joining Status by Candidate Source", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)

        # --- SUB-TAB 3: OFFER FINANCIALS & TIMING ---
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
                    df['Bonus'] = df['Joining Bonus'].apply(lambda x: 'Yes' if str(x)=='1' else ('No' if str(x)=='0' else x))
                    fig = px.histogram(df, x="Bonus", color="Status_Label", barmode="group", 
                                       title="Joining Bonus Offered vs Joining", color_discrete_map=color_map)
                    st.plotly_chart(fig, use_container_width=True)

# --- 5. MODEL COMPARISON TAB ---
elif choice == "Model Comparison":
    st.title("⚙️ Model Performance Comparison")
    st.markdown("This section details the performance of various machine learning algorithms tested during the training phase.")
    
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
    
    
    """)
    st.markdown("---")
    st.markdown("## **Viren Vairagi**")
    st.caption("Internship Project")