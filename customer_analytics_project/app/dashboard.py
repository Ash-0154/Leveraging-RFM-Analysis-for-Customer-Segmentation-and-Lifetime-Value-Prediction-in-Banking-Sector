"""
📊 Customer Analytics Streamlit Dashboard

This frontend application interacts with a FastAPI backend to:
- 🧠 Predict Customer Lifetime Value (CLTV)
- 🧩 Segment customers using clustering models
- 📈 Display interactive visual insights for data-driven decisions

Tech Stack: Streamlit + FastAPI + SHAP + Plotly + AWS S3
"""

import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import os
from PIL import Image
import streamlit.components.v1 as components

# === Page Configuration & Welcome Text ===
# Sets layout, page title, and introductory context for the user.
# Explains key functions like CLTV prediction and customer segmentation.
st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")
st.title("📊 Customer CLTV & Segmentation Dashboard")

st.markdown("""
Welcome to the **Customer Intelligence Platform** powered by Machine Learning.

This tool allows businesses to:
- 🧠 Predict **Customer Lifetime Value (CLTV)**
- 🧩 Segment customers into **behavioral clusters**
- 📈 Understand which features drive customer value

Powered by models trained on real-world banking behavior and transaction data.
""")

# === Sidebar: Input Customer Profile ===
# Collects:
# - Credentials for authentication
# - Behavioral inputs like visits, calls, credit limit
# - Submits input via form to backend FastAPI API
st.sidebar.header("📥 Input Customer Profile")
with st.sidebar.form("form"):
    username = st.text_input("🔐 Username", value="ashika01")
    password = st.text_input("🔐 Password", type="password", value="mlproject0154")

    total_calls = st.number_input("Total Calls Made", 1, 500, 120)
    total_txn = st.number_input("Total Transactions", 1, 100, 20)
    avg_txn_value = st.number_input("Average Transaction Value", 100.0, 100000.0, 12000.0)
    credit_limit = st.slider("Avg Credit Limit", 10000, 200000, 75000)
    visits_bank = st.slider("Bank Visits", 0, 10, 3)
    visits_online = st.slider("Online Visits", 0, 10, 2)
    submitted = st.form_submit_button("🚀 Predict")
# === Payload for API Request ===
# Formats user inputs into the JSON format required by the FastAPI endpoint.
payload = {
    "total_calls_made": total_calls,
    "total_transactions": total_txn,
    "avg_transaction_value": avg_txn_value,
    "Avg_Credit_Limit": credit_limit,
    "Total_visits_bank": visits_bank,
    "Total_visits_online": visits_online
}

# === Main Panel: Prediction Results ===
# On submit, sends a POST request to FastAPI backend.
# Handles:
# - Success response: displays predicted CLTV and assigned customer segment
# - Unauthorized access
# - Connection or server errors
st.markdown("---")
st.header("🔮 Prediction Results")

if submitted:
    try:
        r = requests.post(
            "http://65.1.88.87:8000/predict",
            json=payload,
            auth=HTTPBasicAuth(username, password)
        )
        if r.status_code == 200:
            result = r.json()
            st.success("✅ Prediction complete!")
            st.metric("📈 Predicted CLTV (₹)", f"{result['cltv_prediction']:,.2f}")
            st.metric("🏷️ Customer Segment", f"Cluster {result['segment']}")
        elif r.status_code == 401:
            st.error("❌ Unauthorized: Check your username or password.")
        else:
            st.error("❌ API error. Check FastAPI server or input.")
    except Exception as e:
        st.error(f"⚠️ Could not connect to API: {e}")

with st.expander("ℹ️ What do these predictions mean?", expanded=True):
    st.markdown("""
- **Customer Lifetime Value (CLTV)**: Expected revenue from this customer, predicted using **Histogram Gradient Boosting**.
- **Customer Segment (0–3)**: Behavioral cluster using **ClusterGAN + KMeans**:
    - Cluster 0 → Low value / at-risk
    - Cluster 1 → Loyal, high spenders
    - Cluster 2 → Infrequent but valuable
    - Cluster 3 → Dormant or unpredictable
""")

# === HTML Plot and Image Viewer Functions ===
# Utility functions to:
# - Load and embed local HTML plots (e.g., Plotly, TSNE)
# - Load and show images from the visuals folder
def show_html_plot(file_path, height=400):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=height)

def show_image_row(files, captions, columns=3):
    cols = st.columns(columns)
    for i, file in enumerate(files):
        path = os.path.join("visuals", file)
        if os.path.exists(path):
            with cols[i % columns]:
                st.image(Image.open(path), caption=captions[i], use_container_width=True)

# === Visuals Section ===
st.markdown("---")
st.header("📊 Visual Insights")

# --- KDE Plots
st.subheader("📈 RFM Feature Distributions")
st.markdown("These plots show how **Recency**, **Frequency**, and **Monetary** values are spread across all customers.")

rfm_cols = st.columns(3)
with rfm_cols[0]: show_html_plot("visuals/kde_recency.html", height=430)
with rfm_cols[1]: show_html_plot("visuals/kde_frequency.html", height=430)
with rfm_cols[2]: show_html_plot("visuals/kde_monetary.html", height=430)

# --- RFM Tier Segments
st.subheader("🎯 RFM Segment Count (Low / Medium / High)")
st.image("visuals/rfm_segment_count.png", use_container_width=False, width=500)

# --- Clustering Insights
st.subheader("🧩 Customer Clustering Insights")
clust_cols = st.columns(2)
with clust_cols[0]: show_html_plot("visuals/cluster_distribution.html", height=430)
with clust_cols[1]: show_html_plot("visuals/rfm_radar_chart.html", height=430)

# --- SHAP + Correlation
st.subheader("🔍 Feature Influence on CLTV")
corr = st.columns(2)
with corr[0]: st.image("visuals/shap_summary.png", caption="SHAP Summary", use_container_width=True)
with corr[1]: show_html_plot("visuals/correlation_heatmap.html", height=450)

# --- Model Comparison
st.markdown("---")
st.header("🧪 Model Performance Comparison")

st.markdown("""
Evaluated multiple models for **CLTV prediction** and **Segmentation**:

- **CLTV Models**: MAE, RMSE, R²
- **Segmentation**: Silhouette, CH Index, DB Index
""")

model_cols = st.columns(2)
with model_cols[0]: show_html_plot("visuals/cltv_model_comparison.html", height=450)
with model_cols[1]: show_html_plot("visuals/segmentation_model_comparison.html", height=450)

# === Footer ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.9em; color: gray;'>
Built with ❤️ using <b>FastAPI</b>, <b>Streamlit</b>, <b>Plotly</b>, and your ML pipeline  
<br>© 2025 Ashika S S • All rights reserved
</div>
""", unsafe_allow_html=True)
