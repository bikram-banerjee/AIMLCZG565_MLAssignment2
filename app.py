import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Bank Marketing Classification",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Bank Marketing Classification Model Comparison"}
)

# Custom CSS for better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    /* Font family definitions */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.5px;
    }
    
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #0066cc;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-weight: 800;
    }
    
    h2 {
        color: #0052a3;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        font-weight: 700;
    }
    
    h3 {
        color: #004080;
        font-weight: 700;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #eef2f7 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: #1a1a2e !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #0066cc;
    }
    
    [data-testid="metric-container"] > div {
        color: #1a1a2e !important;
    }
    
    [data-testid="metric-container"] label {
        color: #555 !important;
        font-weight: 500;
    }
    
    [data-testid="metric-container"] p {
        color: #0066cc !important;
        font-weight: 600;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 1.5rem;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        color: #666;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        color: #0066cc;
        background-color: #f0f7ff;
        border-bottom: 3px solid #0066cc;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #0066cc 0%, #004080 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 102, 204, 0.3);
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.5);
        transform: translateY(-2px);
    }
    
    /* Input styling */
    .stSelectbox, .stNumberInput, .stSlider {
        border-radius: 8px;
    }
    
    /* Dataframe styling */
    [data-testid="dataframe"] {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Success and error messages */
    .stSuccess {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 4px;
    }
    
    .stError {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 4px;
    }
    
    .stWarning {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 4px;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin: -1.5rem 0 2rem 0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline(name):
    return joblib.load(f"model/{name}.joblib")

MODELS = [
    "LogisticRegression",
    "DecisionTree",
    "KNN",
    "GaussianNB",
    "RandomForest",
]

def main():
    st.title("🏦 Bank Marketing Classification")
    st.markdown('<p class="subtitle">Predict whether a client will subscribe to a term deposit</p>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666;'><strong>Dataset:</strong> <a href='https://archive.ics.uci.edu/dataset/222/bank+marketing' target='_blank'>UCI Bank Marketing</a></p>",
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(["📊 Metrics", "🔍 Single Prediction", "📁 Batch Prediction"])

    # ------------------------------------------------------------------
    # TAB 1 — METRICS
    # ------------------------------------------------------------------
    with tab1:
        st.header("📊 Evaluation Metrics Comparison")
        st.markdown("Compare the performance of all 5 classification models across key metrics.")
        
        if os.path.exists("model/metrics.json"):
            with open("model/metrics.json") as f:
                metrics = json.load(f)
            dfm = pd.DataFrame(metrics).T
            
            # Display metrics in a nice table with styling
            st.subheader("Performance Metrics Table")
            styled_df = dfm.style.format("{:.4f}")
            
            # Custom function to highlight with readable text
            def highlight_max(val, is_max):
                if is_max:
                    return 'background-color: #e8f0fe; color: #0066cc; font-weight: bold;'
                return ''
            
            def highlight_min(val, is_min):
                if is_min:
                    return 'background-color: #f1f3f4; color: #333333; font-weight: bold;'
                return ''
            
            # Apply highlighting with proper text colors
            max_vals = dfm.idxmax()
            min_vals = dfm.idxmin()
            
            def apply_highlight(row):
                result = []
                for col in row.index:
                    if row.name == max_vals[col]:
                        result.append('background-color: #e8f0fe; color: #0066cc; font-weight: bold;')
                    elif row.name == min_vals[col]:
                        result.append('background-color: #f1f3f4; color: #333333; font-weight: bold;')
                    else:
                        result.append('')
                return result
            
            styled_df = dfm.style.format("{:.4f}").apply(apply_highlight, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Create two columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Accuracy & AUC Comparison")
                chart_data = dfm[["Accuracy", "AUC"]]
                st.bar_chart(chart_data)
            
            with col2:
                st.subheader("Precision, Recall & F1 Comparison")
                st.line_chart(dfm[["Precision", "Recall", "F1"]])
            
            # Display key insights
            best_accuracy = dfm["Accuracy"].idxmax()
            best_auc = dfm["AUC"].idxmax()
            best_f1 = dfm["F1"].idxmax()
            
            st.markdown("---")
            st.subheader("🎯 Key Insights")
            
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            with insight_col1:
                st.metric("Best Accuracy", best_accuracy, f"{dfm.loc[best_accuracy, 'Accuracy']:.4f}")
            with insight_col2:
                st.metric("Best AUC", best_auc, f"{dfm.loc[best_auc, 'AUC']:.4f}")
            with insight_col3:
                st.metric("Best F1 Score", best_f1, f"{dfm.loc[best_f1, 'F1']:.4f}")
        else:
            st.error("⚠️ Metrics not found! Run `python model/train_and_save.py` first to generate metrics.")

    # ------------------------------------------------------------------
    # TAB 2 — SINGLE PREDICTION
    # ------------------------------------------------------------------
    with tab2:
        st.header("🔍 Single Customer Prediction")
        st.markdown("Enter customer details to predict subscription likelihood.")
        
        st.markdown("### Customer Information")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Demographics**")
            age = st.number_input("Age", 18, 95, 30, help="Customer age in years")
            job = st.selectbox(
                "Job",
                ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                 "retired", "self-employed", "services", "student", "technician",
                 "unemployed", "unknown"],
                help="Customer occupation"
            )
            marital = st.selectbox("Marital Status", ["married", "single", "divorced"], help="Current marital status")
            education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"], help="Education level")
            
            st.markdown("**Financial**")
            default = st.selectbox("Credit Default", ["yes", "no"], help="Has credit in default?")
            balance = st.number_input("Yearly Balance (EUR)", -10000, 150000, 1000, help="Average yearly balance")
            housing = st.selectbox("Housing Loan", ["yes", "no"], help="Has housing loan?")
            loan = st.selectbox("Personal Loan", ["yes", "no"], help="Has personal loan?")

        with c2:
            st.markdown("**Contact Information**")
            contact = st.selectbox("Contact Type", ["unknown", "telephone", "cellular"], help="Type of contact")
            day = st.slider("Last Contact Day", 1, 31, 15, help="Day of the month of last contact")
            month = st.selectbox(
                "Last Contact Month",
                ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                help="Month of last contact"
            )
            
            st.markdown("**Campaign Details**")
            duration = st.number_input("Call Duration (sec)", 0, 6000, 300, help="Duration of last contact in seconds")
            campaign = st.number_input("Campaign Contacts", 1, 100, 1, help="Number of contacts in this campaign")
            pdays = st.number_input("Days Since Previous (-1 = none)", -1, 999, -1, help="Number of days since previous contact")
            previous = st.number_input("Previous Contacts", 0, 100, 0, help="Number of previous contacts")
            poutcome = st.selectbox("Previous Outcome", ["unknown", "failure", "other", "success"], help="Outcome of previous campaign")

        input_df = pd.DataFrame(
            [
                {
                    "age": age,
                    "job": job,
                    "marital": marital,
                    "education": education,
                    "default": default,
                    "balance": balance,
                    "housing": housing,
                    "loan": loan,
                    "contact": contact,
                    "day": day,
                    "month": month,
                    "duration": duration,
                    "campaign": campaign,
                    "pdays": pdays,
                    "previous": previous,
                    "poutcome": poutcome,
                }
            ]
        )

        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            selected = st.selectbox("Select Model", MODELS, help="Choose a classification model")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🔮 Make Prediction", use_container_width=True)
        
        if predict_button:
            try:
                pipe = load_pipeline(selected)
                pred = pipe.predict(input_df)[0]
                proba = pipe.predict_proba(input_df)[0]

                st.markdown("---")
                st.markdown("### 📈 Prediction Results")
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(
                        "Prediction",
                        "✅ Will Subscribe" if pred == 1 else "❌ Won't Subscribe",
                        delta=f"{proba[pred]*100:.1f}% confidence"
                    )
                with m2:
                    st.metric("Probability (Subscribe)", f"{proba[1]:.2%}")
                with m3:
                    st.metric("Probability (No Subscribe)", f"{proba[0]:.2%}")

                # Prediction gauge
                st.markdown("### Subscription Probability Gauge")
                pred_percent = proba[1] * 100
                color = "🟢" if pred == 1 else "🔴"
                st.progress(proba[1])
                st.markdown(f"{color} **{pred_percent:.1f}%** likelihood of subscription")
                
                if pred == 1:
                    st.success("✅ **Predicted Outcome:** This customer is likely to subscribe to the term deposit!")
                else:
                    st.warning("⚠️ **Predicted Outcome:** This customer is unlikely to subscribe. Consider targeted incentives.")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}\n\nDid you run the training script? Try: `python model/train_and_save.py`")

    # ------------------------------------------------------------------
    # TAB 3 — BATCH PREDICTION
    # ------------------------------------------------------------------
    with tab3:
        st.header("📁 Batch Prediction")
        st.markdown("Run predictions on multiple customers at once using test_data.csv.")
        
        if os.path.exists("test_data.csv"):
            test_df = pd.read_csv("test_data.csv")
            
            # Summary cards
            st.markdown("### Dataset Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.metric("Total Records", f"{len(test_df):,}")
            with summary_col2:
                if "y" in test_df.columns:
                    subscribe_pct = (test_df["y"].sum() / len(test_df)) * 100
                    st.metric("Subscription Rate", f"{subscribe_pct:.1f}%")
            with summary_col3:
                st.metric("Features", f"{len(test_df.columns) - 1}")
            
            st.markdown("---")
            st.markdown("### Preview of Data")
            st.dataframe(test_df.head(10), use_container_width=True)
            
            st.markdown("---")
            st.markdown("### Run Batch Prediction")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                sel_batch = st.selectbox("Model for Batch Prediction", MODELS, key="batch")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                run_batch = st.button("⚡ Run Batch Prediction", use_container_width=True)
            
            if run_batch:
                with st.spinner("Processing predictions..."):
                    pipe = load_pipeline(sel_batch)
                    X = test_df.drop("y", axis=1)
                    y_true = test_df["y"]

                    preds = pipe.predict(X)
                    proba = pipe.predict_proba(X)[:, 1]

                    out = test_df.copy()
                    out["prediction"] = preds
                    out["prob_subscribe"] = proba
                    
                    # Display results
                    st.markdown("### Prediction Results")
                    st.dataframe(out, use_container_width=True)
                    
                    # Calculate metrics
                    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

                    acc = accuracy_score(y_true, preds)
                    f1 = f1_score(y_true, preds)
                    precision = precision_score(y_true, preds)
                    recall = recall_score(y_true, preds)
                    auc = roc_auc_score(y_true, proba)
                    
                    st.markdown("---")
                    st.markdown("### 📊 Batch Performance Metrics")
                    
                    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                    with metric_col1:
                        st.metric("Accuracy", f"{acc:.4f}")
                    with metric_col2:
                        st.metric("Precision", f"{precision:.4f}")
                    with metric_col3:
                        st.metric("Recall", f"{recall:.4f}")
                    with metric_col4:
                        st.metric("F1 Score", f"{f1:.4f}")
                    with metric_col5:
                        st.metric("AUC", f"{auc:.4f}")
                    
                    # Download results
                    st.markdown("---")
                    csv = out.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("⚠️ test_data.csv not found. Run training first with `python model/train_and_save.py`")

if __name__ == "__main__":
    main()