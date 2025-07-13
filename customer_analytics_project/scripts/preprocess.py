import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# === 1. Load Raw Data ===
def load_data(filepath='data/Credit Card Customer Data.csv'):
    df = pd.read_csv(filepath)
    return df

# === 2. Clean Data ===
def clean_data(df):
    # Drop duplicates
    df = df.drop_duplicates()

    # Round and fix Total_visits_online
    df['Total_visits_online'] = df['Total_visits_online'].round().astype(int)

    # Outlier handling using IQR for selected features
    for col in ['Avg_Credit_Limit', 'Total_visits_online']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)

    return df

# === 3. Feature Engineering (RFM) ===
def create_rfm_features(df):
    # Recency proxy: inverse of call activity (you can adapt this as needed)
    df['Recency'] = 1 / (df['Total_calls_made'] + 1e-6)

    # Frequency: visits to bank + online
    df['Frequency'] = df['Total_visits_bank'] + df['Total_visits_online']

    # Monetary: credit limit (can also use transactions × avg txn value)
    df['Monetary'] = df['Avg_Credit_Limit']

    return df[['Recency', 'Frequency', 'Monetary']]

# === 4. Scale RFM Features ===
def scale_rfm(rfm, save_scaler=False, scaler_path='models/rfm_scaler.pkl'):
    scaler = MinMaxScaler(feature_range=(1, 5))
    rfm_scaled = pd.DataFrame(scaler.fit_transform(rfm), columns=rfm.columns)

    if save_scaler:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"📦 Scaler saved to {scaler_path}")

    return rfm_scaled, scaler

# === 5. Combined Preprocessing ===
def get_preprocessed_rfm(filepath='data/Credit Card Customer Data.csv'):
    df = load_data(filepath)
    df = clean_data(df)
    rfm = create_rfm_features(df)
    rfm_scaled, _ = scale_rfm(rfm)
    return rfm_scaled

# === 6. Main Save Entry ===
if __name__ == '__main__':
    df = load_data()
    df = clean_data(df)
    rfm = create_rfm_features(df)
    rfm_scaled, scaler = scale_rfm(rfm, save_scaler=True)

    print("✅ RFM Features (first 5 rows):")
    print(rfm_scaled.head())
