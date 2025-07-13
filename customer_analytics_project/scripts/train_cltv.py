"""
🧠 CLTV Model Training Script (HistGradientBoostingRegressor)

This script performs the following:
- Loads and cleans customer data
- Constructs RFM-based behavioral features
- Combines them with additional financial features
- Simulates CLTV as target variable
- Trains a Histogram Gradient Boosting Regressor
- Evaluates performance using RMSE and R²
- Saves the model as a .pkl file for API use
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

from preprocess import load_data, clean_data, create_rfm_features

SAVE_PATH = 'models/cltv/hist_gbr_model.pkl'

def prepare_cltv_dataset():
    df = load_data()
    df = clean_data(df)

    # Create RFM features
    rfm = create_rfm_features(df)

    # Merge with other useful features for prediction
    df_model = pd.concat([rfm, df[['Avg_Credit_Limit', 'Total_visits_bank', 'Total_visits_online']].reset_index(drop=True)], axis=1)

    # Simulate CLTV as target — (you can replace this with actual if available)
    df_model['CLTV'] = df['Avg_Credit_Limit'] * (df['Total_visits_bank'] + df['Total_visits_online'])

    return df_model

def train_hist_gradient_boosting(df_model):
    X = df_model.drop('CLTV', axis=1)
    y = df_model['CLTV']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("📊 Evaluation Metrics:")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"  R²:   {r2_score(y_test, y_pred):.4f}")

    return model

if __name__ == '__main__':
    os.makedirs("models/cltv", exist_ok=True)

    print("📦 Preparing dataset for CLTV...")
    df_model = prepare_cltv_dataset()

    print("🧠 Training HistGradientBoostingRegressor...")
    model = train_hist_gradient_boosting(df_model)

    print("💾 Saving model...")
    joblib.dump(model, SAVE_PATH)

    print("✅ CLTV model saved to:", SAVE_PATH)
