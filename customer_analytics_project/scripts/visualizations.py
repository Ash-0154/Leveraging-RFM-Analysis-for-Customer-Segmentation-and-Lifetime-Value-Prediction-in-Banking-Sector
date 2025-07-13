"""
📊 Visualization Generator for Customer Analytics Project

This script generates interactive and static visualizations for:
- RFM distributions
- Customer clusters
- SHAP feature importance
- Model comparisons

Outputs all plots to the `/visuals` folder in `.html` and `.png` formats.
"""
import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from preprocess import get_preprocessed_rfm, load_data, clean_data, create_rfm_features

# Ensure visuals folder exists
os.makedirs("visuals", exist_ok=True)

# === 1. KDE Plotly Histograms ===
def plot_rfm_kde(rfm):
    for col in rfm.columns:
        fig = px.histogram(rfm, x=col, nbins=40, marginal="box", opacity=0.7,
                           title=f"Distribution of {col}", template="plotly_white")
        fig.write_html(f"visuals/kde_{col.lower()}.html")
        fig.write_image(f"visuals/kde_{col.lower()}.png")
    print("✅ KDE histograms saved.")

# === 2. Correlation Heatmap ===
def plot_correlation_heatmap(df):
    df_num = df.select_dtypes(include=[np.number])
    corr = df_num.corr().round(2)
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap",
                    color_continuous_scale="RdBu_r", template="plotly_white")
    fig.write_html("visuals/correlation_heatmap.html")
    fig.write_image("visuals/correlation_heatmap.png")
    print("✅ Correlation heatmap saved.")

# === 3. Cluster Count Plot ===
def plot_cluster_distribution(labels):
    cluster_df = pd.DataFrame({'Cluster': labels})
    fig = px.histogram(cluster_df, x='Cluster', template="plotly_white", title="Cluster Distribution")
    fig.write_html("visuals/cluster_distribution.html")
    fig.write_image("visuals/cluster_distribution.png")
    print("✅ Cluster distribution saved.")

# === 4. Radar Chart ===
def plot_radar_chart(rfm, labels):
    rfm_labeled = rfm.copy()
    rfm_labeled["Cluster"] = labels
    cluster_stats = rfm_labeled.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().reset_index()

    categories = ['Recency', 'Frequency', 'Monetary']
    fig = go.Figure()

    for _, row in cluster_stats.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[c] for c in categories] + [row[categories[0]]],
            theta=categories + [categories[0]],
            fill='toself',
            name=f"Cluster {int(row['Cluster'])}"
        ))

    fig.update_layout(
        title="RFM Radar Chart by Cluster",
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        template="plotly_white"
    )
    fig.write_html("visuals/rfm_radar_chart.html")
    fig.write_image("visuals/rfm_radar_chart.png")
    print("✅ Radar chart saved.")

def plot_shap_summary():
    import matplotlib.pyplot as plt
    model = joblib.load("models/cltv/hist_gbr_model.pkl")
    df = clean_data(load_data())
    rfm = create_rfm_features(df)
    features = pd.concat([
        rfm,
        df[['Avg_Credit_Limit', 'Total_visits_bank', 'Total_visits_online']].reset_index(drop=True)
    ], axis=1)

    explainer = shap.Explainer(model)
    shap_values = explainer(features)

    # Increase figure size before saving
    plt.figure(figsize=(7, 6))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Summary Plot - CLTV Model", fontsize=14)
    plt.tight_layout()
    plt.savefig("visuals/shap_summary.png", dpi=300)
    plt.close()
    print("✅ SHAP plot saved.")

# === 6. RFM Tier Segments ===
def plot_rfm_segment_counts(rfm):
    seg_r = pd.qcut(rfm['Recency'], 3, labels=["High", "Medium", "Low"])
    seg_f = pd.qcut(rfm['Frequency'], 3, labels=["Low", "Medium", "High"])
    seg_m = pd.qcut(rfm['Monetary'], 3, labels=["Low", "Medium", "High"])

    df_seg = pd.DataFrame({
        "Recency": seg_r,
        "Frequency": seg_f,
        "Monetary": seg_m
    })

    melted = df_seg.melt(var_name="RFM_Feature", value_name="Segment")
    grouped = melted.value_counts().reset_index(name="Count")

    fig = px.bar(grouped, x="Segment", y="Count", color="RFM_Feature", barmode="group",
                 title="RFM Segment Distribution", template="plotly_white")
    fig.write_html("visuals/rfm_segment_count.html")
    fig.write_image("visuals/rfm_segment_count.png")
    print("✅ RFM segment plot saved.")

# === 7. CLTV Model Comparison (Hardcoded) ===
def add_model_comparison_cltv():
    models = ["XGBoost", "LightGBM", "ExtraTrees", "HistGBR", "Ridge", "Lasso", "NGBoost"]
    mae = [3733.88, 20243.79, 6964.23, 5657.61, 35757.19, 35847.82, 3625.59]
    rmse = [8239.28, 36166.23, 9677.36, 11487.35, 43809.50, 43917.64, 8460.21]
    r2 = [0.9987, 0.9741, 0.9981, 0.9974, 0.9620, 0.9618, 0.9986]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="MAE", x=models, y=mae))
    fig.add_trace(go.Bar(name="RMSE", x=models, y=rmse))
    fig.add_trace(go.Bar(name="R²", x=models, y=r2))

    fig.update_layout(
        title="CLTV Model Comparison",
        barmode='group',
        template="plotly_white"
    )
    fig.write_html("visuals/cltv_model_comparison.html")
    fig.write_image("visuals/cltv_model_comparison.png")
    print("✅ CLTV comparison saved.")

# === 8. Segmentation Model Comparison (Hardcoded) ===
def add_model_comparison_segmentation():
    models = ["KMeans", "GMM", "HDBSCAN", "DEC", "SOM", "SpectralNet", "ClusterGAN"]
    silhouette = [0.7306, 0.3176, 0.1926, 0.8952, 0.0312, 0.5997, 0.8791]
    calinski = [5395.36, 366.55, 105.82, 2349.83, 59.16, 2635.93, 3369.38]
    db_index = [0.3838, 1.1097, 1.4529, 0.3513, 1.3916, 0.5295, 0.3116]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Silhouette", x=models, y=silhouette))
    fig.add_trace(go.Bar(name="Calinski-Harabasz", x=models, y=calinski))
    fig.add_trace(go.Bar(name="Davies-Bouldin", x=models, y=db_index))

    fig.update_layout(
        title="Segmentation Model Comparison",
        barmode='group',
        template="plotly_white"
    )
    fig.write_html("visuals/segmentation_model_comparison.html")
    fig.write_image("visuals/segmentation_model_comparison.png")
    print("✅ Segmentation comparison saved.")

# === 9. Run All ===
def generate_all_visuals():
    print("📈 Loading and preprocessing data...")
    rfm = get_preprocessed_rfm()
    df = clean_data(load_data())

    print("📊 Generating visualizations...")
    plot_rfm_kde(rfm)
    plot_correlation_heatmap(df)
    plot_rfm_segment_counts(rfm)

    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(rfm)

    plot_cluster_distribution(labels)
    plot_radar_chart(rfm, labels)
    plot_shap_summary()
    add_model_comparison_cltv()
    add_model_comparison_segmentation()

    print("🎉 All visualizations saved in /visuals")

# Run script
if __name__ == '__main__':
    generate_all_visuals()
