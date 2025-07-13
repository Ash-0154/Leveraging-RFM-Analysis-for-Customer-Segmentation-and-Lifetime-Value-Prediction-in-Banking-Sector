# Leveraging RFM Analysis for Customer Segmentation and Lifetime Value Prediction in the Banking Sector


A full-stack machine learning platform that leverages RFM analysis to predict Customer Lifetime Value (CLTV) and segment customers using advanced clustering techniques. Designed for the banking and financial services sector, this project enables data-driven targeting and customer management through interactive dashboards and a secure cloud deployment.

---
## Problem Statement

Banking institutions often face challenges in identifying high-value customers and delivering personalized engagement strategies. This project aims to solve this by:

- Predicting **Customer Lifetime Value (CLTV)** using behavioral data.
- Segmenting customers based on **RFM metrics** (Recency, Frequency, Monetary).
- Delivering real-time insights via an interactive web dashboard.

---
## Dataset
Source: Kaggle - Credit Card Customer Dataset

Fields include: Avg_Credit_Limit, Total_Visits_Online, Total_Credit_Cards, Total_Calls_Made, etc.

Used for both CLTV prediction and segmentation

---

## Solution Overview

| Component        | Description                                                                |
|------------------|----------------------------------------------------------------------------|                 
| RFM Engineering  | Derives behavioral features from customer interaction data                 |
| ML Models        | Ensemble-based CLTV regression and clustering for customer segmentation    |
| Visualization    | Streamlit dashboard for real-time insights                                 |
| Deployment       | FastAPI + Streamlit hosted on AWS EC2                                      |
| Security         | Includes Basic Auth, Rate Limiting, S3 backup                              |

## Architecture
```
User Input ─► Streamlit UI
│
HTTP Request to /predict
▼
FastAPI Backend ─► CLTV + Segmentation Models
▼
Prediction Logs (CSV + S3)
```
---

## Feature Engineering

| Feature   | Description                                                     |
|-----------|-----------------------------------------------------------------|
| Recency   | `1 / (total_calls_made + ε)` – how recently customer interacted |
| Frequency | Total online + bank visits                                      |
| Monetary  | `Avg_Credit_Limit` – proxy for customer value                   |

All RFM features are normalized using `MinMaxScaler`.

---
## CLTV Prediction

### Best Model: HistGradientBoosting Regressor

### Final Model Performance

| Model                    | MAE     | RMSE    | R² Score  |
|--------------------------|---------|---------|-----------|
| HistGradientBoosting     | 3144.99 | 7126.28 | **0.9991**|
| Extra Trees              | 3363.13 | 7464.70 | 0.9989    |
| XGBoost                  | 3733.89 | 8239.28 | 0.9987    |
| LightGBM                 | 3774.82 | 8369.67 | 0.9986    |
| NGBoost                  | 3625.59 | 8460.22 | 0.9986    |
| Ridge / Lasso            | ~35700  | ~43900  | 0.9620    |


---

## Customer Segmentation

### Best Model: ClusterGAN + KMeans

| Model        | Silhouette | CH Index | DB Index  |
|--------------|------------|----------|-----------|
| ClusterGAN   | **0.8791** | **3369** | **0.311** |
| DEC          | 0.7207     | 2580.54  | 0.4262    |
| Spectral Net | 0.5997     | 2635.93  | 0.5295    |
| K-Means      | 0.5216     | 1047.61  | 0.6511    |
| GMM          | 0.4882     | 943.09   | 0.6797    |
| HDBSCAN      | 0.1926     | 105.82   | 1.4529    |
| SOM          | 0.0312     | 59.16    | 1.3916    |

---
## Dashboard Features- Streamlit

- Real-time CLTV prediction via user input sliders.
- Cluster assignment with explanations.
- SHAP value plots for feature importance.
- Model comparison, radar charts, KDE plots.
<img width="1915" height="822" alt="Screenshot 2025-07-13 015150" src="https://github.com/user-attachments/assets/1b6ff9a2-f3e2-4dc8-b60c-311050934c2d" />
<img width="1919" height="832" alt="Screenshot 2025-07-13 015205" src="https://github.com/user-attachments/assets/a618e0e5-fd0d-41b8-9543-9a7afcafc4eb" />
<img width="1588" height="736" alt="Screenshot 2025-07-12 011444" src="https://github.com/user-attachments/assets/f935fda7-dbe6-4005-8339-1b58b30c0bfd" />
<img width="1604" height="565" alt="Screenshot 2025-07-12 011501" src="https://github.com/user-attachments/assets/6ad35065-5b61-4866-a8bc-731b96aa2ba6" />
<img width="1584" height="635" alt="Screenshot 2025-07-12 011517" src="https://github.com/user-attachments/assets/45bf8958-be09-483e-b3c7-42093b0fbd3b" />
<img width="1625" height="742" alt="Screenshot 2025-07-12 011533" src="https://github.com/user-attachments/assets/201fda56-ad19-4fcc-9ab4-5117d2b782eb" />
<img width="1660" height="872" alt="Screenshot 2025-07-12 011600" src="https://github.com/user-attachments/assets/bfb2200c-2508-4ecf-9fa4-50816c844d14" />
<img width="1883" height="792" alt="Screenshot 2025-07-13 015255" src="https://github.com/user-attachments/assets/88e1eb94-c307-4ed0-8283-bc5322d9b88e" />

---

## Deployment Guide
### Project Structure
```
customer-analytics-project/
├── app/
│   ├── api.py              # FastAPI backend
│   └── dashboard.py        # Streamlit UI
├── models/
│   ├── cltv/               # Saved regression models
│   └── segmentation/       # Saved clustering models
├── scripts/
│   ├── train_cltv.py
│   ├── train_segmentation.py
│   └── preprocess.py
├── data/                   # Input datasets
├── logs/                   # Prediction logs
├── visuals/                # Plots, figures
├── requirements.txt
└── README.md
```


1. **EC2 Setup**  
   Launched a `t2.micro` instance (Ubuntu 22.04)  
   Open ports: `22`, `8000`, `8501`

2. **Upload & SSH**
   ```bash
   scp -i "key.pem" -r customer-analytics-project ubuntu@<elastic-ip>:
   ssh -i "key.pem" ubuntu@<elastic-ip>
   
3. **Install & Setup**
    ```bash
    sudo apt update && sudo apt install python3-venv
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    
4. **Run Services**
    ```bash
    nohup uvicorn app.api:app --host 0.0.0.0 --port 8000 &
    nohup streamlit run app/dashboard.py --server.address 0.0.0.0 --server.port 8501 &
5.**Optional: Store Prediction Logs in S3**
- Configure IAM credentials using `aws configure`
- Bucket Name: `customer-predictions`
- Code uses `boto3` to upload `logs/predictions.csv` automatically on each POST request

 Security & Cloud Features
- HTTP Basic Authentication added to protect the /predict endpoint using a secure username/password scheme (via FastAPI).
- Rate Limiting implemented using the slowapi library to allow only 10 requests per minute per IP address.
- Logging Predictions: Each prediction request is logged to a local logs/predictions.csv file for audit and traceability.
- S3 Cloud Integration: The logged predictions are automatically uploaded to an AWS S3 bucket (customer-predictions) after every API call.
- Deployed on AWS EC2: The full stack (FastAPI + Streamlit + models) is hosted on a Free Tier EC2 instance with Elastic IP for persistent access.
- Secure SSH Access: Instance protected via private key authentication.

---

## Future Work
- Add time-series CLTV modeling
- Expand to include demographic and transaction-level features
- Add XAI explainability layers with LIME/SHAP dashboards
- Streamlit Cloud or AWS ECS deployment
- Scheduled retraining (cron + Lambda)

