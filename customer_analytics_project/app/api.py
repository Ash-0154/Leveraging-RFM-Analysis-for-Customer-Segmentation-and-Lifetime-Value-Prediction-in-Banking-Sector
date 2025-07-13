"""
📦 Customer Analytics API (CLTV Prediction & Segmentation)

This FastAPI application provides a secure and rate-limited endpoint `/predict`
that accepts customer behavior features and returns:
- Predicted Customer Lifetime Value (CLTV)
- Cluster Segment Label (from segmentation model)

Key Features:
- HTTP Basic Authentication
- Request rate limiting (10 requests/min per IP)
- Logging of each prediction to local CSV + AWS S3 bucket
"""
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import joblib
import pandas as pd
import numpy as np
import csv
import os
from datetime import datetime
import boto3
from botocore.exceptions import NoCredentialsError

# === Load Pretrained Models ===
# - CLTV Regressor: HistogramGradientBoostingRegressor (hist_gbr_model.pkl)
# - Customer Segmentation: ClusterGAN Model
# - RFM Scaler: MinMaxScaler fitted on Recency, Frequency, Monetary
cltv_model = joblib.load("models/cltv/hist_gbr_model.pkl")
segmentation_model = joblib.load("models/segmentation/clustergan_model.pkl")
rfm_scaler = joblib.load("models/rfm_scaler.pkl")

# === FastAPI App Setup ===
# Initializes FastAPI app with SlowAPI limiter for rate limiting.
# Allows only 10 prediction requests per minute per IP address.
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

security = HTTPBasic()

# === HTTP Basic Authentication ===
# Uses fixed credentials (username/password) to protect the /predict route.
# Update VALID_USERNAME and VALID_PASSWORD as needed.
VALID_USERNAME = "USERNAME"
VALID_PASSWORD = "PASSWORD"

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != VALID_USERNAME or credentials.password != VALID_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

# === Input Schema ===
class CustomerFeatures(BaseModel):
    total_calls_made: int
    total_transactions: int
    avg_transaction_value: float
    Avg_Credit_Limit: float
    Total_visits_bank: int
    Total_visits_online: int

# === Upload to S3 Helper ===
# Uploads logs/predictions.csv to the given S3 bucket.
# Requires AWS credentials to be configured using `aws configure`.
def upload_to_s3(local_path, bucket_name, s3_key):
    try:
        s3 = boto3.client('s3')
        s3.upload_file(local_path, bucket_name, s3_key)
        print(f"✅ Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
    except FileNotFoundError:
        print("❌ Log file not found for upload.")
    except NoCredentialsError:
        print("❌ AWS credentials not found. Run aws configure.")

# === POST /predict ===
# Accepts customer features, applies RFM logic and scaling,
# Predicts CLTV and customer segment,
# Logs the result locally and to AWS S3.
# Secured via Basic Auth and rate-limited to 10 req/min.
"""
Calculates:
- Recency = 1 / (Total Calls Made + small epsilon)
- Frequency = Total Visits Bank + Total Visits Online
- Monetary = Avg Credit Limit

Pipeline:
1. Scale RFM values using saved MinMaxScaler
2. Create full feature set for CLTV prediction
3. Predict CLTV using the regression model
4. Predict segment using clustering model
5. Log input + output to CSV
6. Upload log to AWS S3
"""

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, features: CustomerFeatures, credentials: HTTPBasicCredentials = Depends(authenticate)):
    recency = 1 / (features.total_calls_made + 1e-6)
    frequency = features.Total_visits_bank + features.Total_visits_online
    monetary = features.Avg_Credit_Limit

    rfm_df = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary
    }])

    rfm_scaled = pd.DataFrame(
        rfm_scaler.transform(rfm_df),
        columns=["Recency", "Frequency", "Monetary"]
    )

    cltv_input = pd.concat([
        rfm_scaled,
        pd.DataFrame([{
            "Avg_Credit_Limit": features.Avg_Credit_Limit,
            "Total_visits_bank": features.Total_visits_bank,
            "Total_visits_online": features.Total_visits_online
        }])
    ], axis=1)

    cltv_prediction = cltv_model.predict(cltv_input)[0]
    segment_input = rfm_scaled.to_numpy().astype(np.float64)
    segment_label = int(segmentation_model.predict(segment_input)[0])

    # === Logging ===
    # Appends predictions and input values to logs/predictions.csv.
    # File is auto-created with headers if it doesn't exist.
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/predictions.csv"
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "Avg_Credit_Limit": features.Avg_Credit_Limit,
        "Total_visits_bank": features.Total_visits_bank,
        "Total_visits_online": features.Total_visits_online,
        "predicted_cltv": round(float(cltv_prediction), 2),
        "segment": segment_label
    }

    write_header = not os.path.exists(log_path)
    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_entry.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(log_entry)

    # === S3 Upload ===
    # Automatically uploads the prediction log to S3 after every request.
    # Ensures backup and availability of logs in cloud.

    upload_to_s3(
        local_path=log_path,
        bucket_name="customer-analytics-ml-project",  
        s3_key="logs/predictions.csv"
    )

    return {
        "cltv_prediction": round(float(cltv_prediction), 2),
        "segment": segment_label
    }
