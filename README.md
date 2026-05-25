# 🎬 StreamIQ — OTT Churn Prediction & Retention Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.0+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

**A production-grade ML analytics platform that predicts OTT subscriber churn and delivers AI-powered retention strategies — built with a Netflix-inspired premium dark UI.**

[Live Demo](#deployment) · [Features](#features) · [Installation](#installation) · [Architecture](#architecture)

</div>

---

## 📸 Screenshots

> _Dashboard previews coming soon — run the app locally to see the full experience._

| Executive Dashboard | EDA Analytics | Churn Predictor |
|---|---|---|
| ![Dashboard](assets/dashboard.png) | ![Analytics](assets/analytics.png) | ![Predictor](assets/predictor.png) |

---

## 🎯 Project Overview

StreamIQ is an end-to-end **Customer Churn Prediction & Retention Intelligence System** designed for OTT (Over-The-Top) streaming platforms. It combines advanced machine learning with a stunning, recruiter-ready analytics dashboard to:

- 🔮 **Predict** which subscribers are likely to cancel
- 📊 **Analyse** platform usage, engagement, and revenue patterns
- 🎯 **Segment** customers by churn risk (Low / Medium / High)
- 💡 **Recommend** personalised, data-driven retention strategies

---

## ✨ Features

### 🏠 Executive Dashboard
- 6 animated KPI cards: Total Customers, Churn Rate, Revenue at Risk, Watch Hours, High-Risk Users, Retention Rate
- Churn distribution donut chart
- Churn rate by account age
- Watch hours distribution by churn status
- Subscription tier analysis

### 📊 EDA Analytics
- Interactive filters (subscription type, gender, age range)
- Genre-level churn analysis
- Binge watch frequency vs churn correlation
- Revenue breakdown (safe vs at-risk)
- Payment method churn distribution
- Session time scatter plots
- Days-since-login trend analysis
- Feature correlation heatmap

### 🤖 ML Model Comparison
- 4 algorithms: Logistic Regression, Decision Tree, Random Forest, XGBoost
- Full metrics table: Accuracy, Precision, Recall, F1, ROC-AUC
- Interactive ROC curves (all models overlaid)
- Radar chart model comparison
- Feature importance visualisation (top 12 predictive features)

### 🔮 Churn Predictor
- Beautiful customer profile input form
- Real-time churn probability scoring
- Animated gauge chart
- Risk badge (Low / Medium / High)
- Personalised retention recommendations

### 💡 Business Intelligence
- Risk segmentation KPIs
- Revenue by risk segment (pie chart)
- Risk distribution by subscription tier
- Top 20 high-risk customer table
- 6 strategic retention recommendations with business rationale

---

## 🏗️ Architecture

```
OTT-Churn-Prediction/
│
├── data/
│   ├── generate_data.py          # Synthetic dataset generator (5,000 rows)
│   └── ott_churn_data.csv        # Generated dataset
│
├── src/
│   ├── preprocessing.py          # Data cleaning, encoding, scaling
│   └── model_training.py         # Model training, evaluation, persistence
│
├── models/                       # Persisted ML artifacts (joblib)
│   ├── best_model.pkl
│   ├── all_results.pkl
│   ├── feature_importance.pkl
│   ├── encoders.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── app/
│   └── streamlit_app.py          # Main Streamlit dashboard (5 pages)
│
├── assets/                       # Screenshots / brand assets
├── notebooks/                    # EDA notebooks (optional)
│
├── .streamlit/
│   └── config.toml               # Streamlit theme config
│
├── train.py                      # One-shot training pipeline
├── requirements.txt
└── README.md
```

---

## 🤖 ML Workflow

```
Raw Data (5,000 OTT customers)
        │
        ▼
  Data Generation (19 features, realistic churn logic)
        │
        ▼
  Preprocessing
  ├── Missing value imputation (median strategy)
  ├── Label encoding (7 categorical features)
  ├── StandardScaler normalisation
  └── 80/20 stratified train-test split
        │
        ▼
  Model Training & Evaluation
  ├── Logistic Regression
  ├── Decision Tree (max_depth=6)
  ├── Random Forest (200 estimators)
  └── XGBoost (200 estimators, lr=0.05)
        │
        ▼
  Auto Best Model Selection (by ROC-AUC)
        │
        ▼
  Artifact Persistence (joblib)
        │
        ▼
  Streamlit Dashboard
  ├── Executive KPIs
  ├── EDA Visualisations
  ├── Model Comparison
  ├── Real-time Prediction API
  └── Business Intelligence
```

### Dataset Features

| Feature | Type | Description |
|---|---|---|
| `customer_id` | ID | Unique customer identifier |
| `age` | Numeric | Customer age (18–70) |
| `gender` | Categorical | Male / Female / Other |
| `subscription_type` | Categorical | Basic / Standard / Premium |
| `monthly_watch_hours` | Numeric | Hours watched per month |
| `favorite_genre` | Categorical | Action / Drama / Comedy etc. |
| `devices_used` | Numeric | Number of streaming devices |
| `avg_session_time` | Numeric | Average session length (hours) |
| `monthly_subscription_cost` | Numeric | Monthly plan cost (₹) |
| `payment_method` | Categorical | Credit Card / UPI / Wallet etc. |
| `support_tickets` | Numeric | Open support tickets |
| `ads_tolerance` | Categorical | Low / Medium / High |
| `days_since_last_login` | Numeric | Inactivity indicator |
| `number_of_profiles` | Numeric | Sub-profiles on account |
| `binge_watch_frequency` | Categorical | Never → Always scale |
| `content_rating_given` | Numeric | Average rating (1–5) |
| `account_age_months` | Numeric | Tenure in months |
| `auto_renew_enabled` | Binary | Auto-renewal toggle |
| `churn` | Target | 0 = Retained, 1 = Churned |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/OTT-Churn-Prediction.git
cd OTT-Churn-Prediction

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models (one-time setup, ~30–60 seconds)
python train.py

# 5. Launch the dashboard
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

> **Note:** If you skip step 4, the dashboard will auto-train models on first launch.

---

## 🚀 Deployment

### Streamlit Cloud
1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Set **Main file path**: `app/streamlit_app.py`
4. Deploy — Streamlit Cloud will auto-install `requirements.txt`

### Render
```yaml
# render.yaml
services:
  - type: web
    name: streamiq-churn
    env: python
    buildCommand: pip install -r requirements.txt && python train.py
    startCommand: streamlit run app/streamlit_app.py --server.port $PORT
```

### Hugging Face Spaces
```
# Set SDK: Streamlit
# App file: app/streamlit_app.py
# Add requirements.txt at repo root
```

---

## 📈 Model Performance

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.833 | 0.046 | **0.694** |
| Decision Tree | 0.827 | 0.188 | 0.639 |
| Random Forest | 0.835 | 0.088 | 0.688 |
| XGBoost | 0.828 | 0.122 | 0.655 |

_Best model selected automatically by ROC-AUC._

---

## 🔮 Future Improvements

- [ ] Add SHAP explainability visualisations
- [ ] Implement cohort retention analysis
- [ ] Add time-series churn forecasting (LSTM)
- [ ] REST API endpoint for real-time scoring
- [ ] A/B test simulation for retention campaigns
- [ ] Multi-region deployment with AWS/GCP
- [ ] User authentication and role-based access
- [ ] Automated model retraining pipeline

---

## 📋 Resume Description

> **StreamIQ — OTT Churn Prediction & Retention Intelligence System** _(Personal Project)_
>
> Built a production-grade ML analytics platform to predict OTT subscriber churn using Python, XGBoost, and Streamlit. Engineered a 5,000-row synthetic dataset with 19 features and realistic churn logic. Trained and compared 4 classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost) achieving 83.5% accuracy and 0.694 ROC-AUC. Designed a premium Netflix-inspired dark-mode Streamlit dashboard with 5 interactive pages, Plotly visualisations, real-time churn scoring, risk segmentation, and AI-powered retention recommendations.
>
> **Tech:** Python · Pandas · NumPy · Scikit-learn · XGBoost · Plotly · Streamlit · Joblib

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Engineering | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualisation | Plotly, Seaborn, Matplotlib |
| Dashboard | Streamlit + Custom CSS |
| Persistence | Joblib |
| Deployment | Streamlit Cloud / Render / HF Spaces |

---

<div align="center">
  Made with ❤️ for portfolio & placement purposes
</div>
