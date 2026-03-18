import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add scripts to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from anomaly_detection import AnomalyDetection
from data_ingestion import DataIngestion

# Set page config
st.set_page_config(
    page_title="Financial Anomaly Detection System",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">💰 Financial Anomaly Detection System</h1>', unsafe_allow_html=True)
    st.markdown("""
    This system detects anomalous transactions in financial data using machine learning algorithms.
    Upload your data or use the sample dataset to see the anomaly detection in action.
    """)

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Sample Data", "Upload CSV"]
    )

    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Anomaly Detection Algorithm:",
        ["Isolation Forest", "DBSCAN"]
    )

    # Parameters
    contamination = st.sidebar.slider(
        "Contamination Rate (expected % of anomalies):",
        0.01, 0.5, 0.1, 0.01
    )

    eps = st.sidebar.slider(
        "DBSCAN EPS (neighborhood distance):",
        0.1, 2.0, 0.5, 0.1
    ) if algorithm == "DBSCAN" else None

    # Load data
    if data_source == "Sample Data":
        if st.sidebar.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                # Import and run sample data generation
                from generate_sample_data import generate_sample_data
                df = generate_sample_data(n=1000)
                st.session_state['data'] = df
                st.success("Sample data generated!")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state['data'] = df
            st.success("Data uploaded successfully!")

    # Check if data exists
    if 'data' not in st.session_state:
        st.info("Please generate sample data or upload a CSV file to get started.")
        return

    df = st.session_state['data']

    # Data Overview
    st.header("📊 Data Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Transactions", len(df))

    with col2:
        st.metric("Total Amount", f"${df['Amount'].sum():,.2f}")

    with col3:
        st.metric("Average Amount", f"${df['Amount'].mean():,.2f}")

    with col4:
        risk_pct = (df['Risk_Flag'].sum() / len(df)) * 100
        st.metric("Known Risk %", f"{risk_pct:.1f}%")

    # Data preview
    st.subheader("Sample Transactions")
    st.dataframe(df.head(10), use_container_width=True)

    # Data visualization
    st.header("📈 Data Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Amount Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df['Amount'], bins=50, ax=ax)
        ax.set_title("Transaction Amount Distribution")
        ax.set_xlabel("Amount ($)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with col2:
        st.subheader("Transactions by Department")
        fig, ax = plt.subplots(figsize=(8, 6))
        df['Department'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Transactions by Department")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Anomaly Detection
    st.header("🔍 Anomaly Detection Results")

    if st.button("Run Anomaly Detection", type="primary"):
        with st.spinner("Running anomaly detection..."):
            # Prepare data
            ingestion = DataIngestion()
            df_processed = ingestion.handle_missing_values(df.copy())
            df_processed = ingestion.feature_engineering(df_processed)

            # Select features for detection
            numeric_cols = ['Amount', 'Amount_Log']
            if 'Day_of_Week' in df_processed.columns:
                numeric_cols.append('Day_of_Week')

            # Scale data
            scaler = StandardScaler()
            X = scaler.fit_transform(df_processed[numeric_cols].values)

            # Run detection
            detector = AnomalyDetection()

            if algorithm == "Isolation Forest":
                preds = detector.detect_iforest(X)
                scores = detector.train_iforest(X)
                anomaly_scores = scores
            else:  # DBSCAN
                preds = detector.detect_dbscan(X)
                anomaly_scores = None

            # Add results to dataframe
            df_results = df.copy()
            df_results['Anomaly_Prediction'] = preds
            df_results['Is_Anomaly'] = (preds == -1)

            # Display results
            st.subheader("Detection Summary")
            col1, col2, col3 = st.columns(3)

            anomalies_detected = df_results['Is_Anomaly'].sum()
            normal_transactions = len(df_results) - anomalies_detected

            with col1:
                st.metric("Anomalies Detected", anomalies_detected)

            with col2:
                st.metric("Normal Transactions", normal_transactions)

            with col3:
                detection_rate = (anomalies_detected / len(df_results)) * 100
                st.metric("Detection Rate", f"{detection_rate:.1f}%")

            # Anomalies table
            st.subheader("Detected Anomalies")
            anomalies_df = df_results[df_results['Is_Anomaly']].head(20)
            st.dataframe(anomalies_df[['Transaction_ID', 'Vendor', 'Amount', 'Department', 'Description']], use_container_width=True)

            # Visualization of anomalies
            st.subheader("Anomaly Visualization")

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['red' if x else 'blue' for x in df_results['Is_Anomaly']]
                scatter = ax.scatter(df_results['Amount'], range(len(df_results)), c=colors, alpha=0.6)
                ax.set_xlabel("Amount ($)")
                ax.set_ylabel("Transaction Index")
                ax.set_title("Anomalies by Amount")
                ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Anomaly'),
                                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Normal')])
                st.pyplot(fig)

            with col2:
                if anomaly_scores is not None:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.histplot(anomaly_scores, bins=50, ax=ax)
                    ax.set_title("Anomaly Score Distribution")
                    ax.set_xlabel("Anomaly Score")
                    ax.set_ylabel("Frequency")
                    ax.axvline(x=np.percentile(anomaly_scores, (1-contamination)*100), color='red', linestyle='--', label='Threshold')
                    ax.legend()
                    st.pyplot(fig)

            # Download results
            st.subheader("Download Results")
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="Download Full Results as CSV",
                data=csv,
                file_name="anomaly_detection_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()