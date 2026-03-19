# Customer Churn Prediction System

A complete end-to-end machine learning system that predicts telecom customer churn 30 days in advance, enabling proactive retention campaigns.

**[Live Demo](https://customer-churn-prediction-telecom.streamlit.app)** | **[Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)**

---

## Business Problem

Telecom companies lose 15-25% of customers annually. Each lost customer represents ~$1,800 in lifetime value. Identifying at-risk customers before they churn enables targeted retention campaigns that are far cheaper than acquiring new customers.

## Solution

An XGBoost classification model that identifies 76.7% of churners 30 days in advance, deployed as an interactive web application with real-time risk scoring.

---

## Business Impact

| Metric | Value |
|--------|-------|
| Annual Revenue Protected | $637,850 |
| Campaign ROI | 468% |
| Customers Saved Per Year | 430 |
| Churners Identified | 76.7% of at-risk customers |
| Early Warning Window | 30 days before churn |

---

## Key Findings

- **Contract type is the strongest predictor** — Month-to-month customers churn at 42.7% vs 2.8% for two-year contracts (15x difference)
- **New high-paying customers are highest risk** — `charges_to_tenure_ratio` was the #1 SHAP feature, a custom-engineered variable
- **Fiber optic customers churn at 41.9%** — pricing vs perceived value issue
- **Electronic check payment = 43% churn** — correlates strongly with month-to-month contracts
- **Customers without OnlineSecurity or TechSupport churn at 40%+** vs 15% with these services

---

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost |
| ROC-AUC | 0.8495 |
| Recall @ threshold 0.3 | 76.7% |
| Precision @ threshold 0.3 | 52.7% |
| Cross-validation Std | 0.011 (stable) |
| Training samples | 5,634 |
| Features | 34 (including 4 engineered) |

**Why threshold 0.3?** The cost of missing a churner (~$1,800 CLV lost) far outweighs the cost of a false alarm (~$50 retention offer). A lower threshold catches more churners at the expense of some false positives — the right tradeoff for this business problem.

---

## Engineered Features

Standard churn models use raw data. This model adds four business-driven features that capture insights the algorithm wouldn't find on its own:

| Feature | Business Logic |
|---------|---------------|
| `charges_to_tenure_ratio` | New customers paying high amounts = high risk. Became the #1 SHAP predictor |
| `service_adoption_score` | Counts active services. Low engagement predicts disengagement |
| `high_risk_combo` | Flags month-to-month + charges > $65. Encodes the highest-risk profile directly |
| `tenure_group` | Bins tenure into New/Developing/Established/Loyal loyalty stages |

---

## Project Structure
```
customer-churn-prediction/
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb # Feature creation and encoding
│   ├── 03_modeling.ipynb            # Model training and evaluation
│   └── 04_business_impact.ipynb     # ROI and business case analysis
│
├── models/
│   ├── churn_model.pkl              # Trained XGBoost model
│   └── scaler.pkl                   # StandardScaler for numerical features
│
├── reports/                         # Charts and visualizations
├── app.py                           # Streamlit dashboard
├── requirements.txt
└── packages.txt
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data Processing | Pandas, NumPy |
| Modeling | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

---

## Run Locally
```bash
# Clone the repository
git clone https://github.com/SubramaniMokkala/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Go to https://www.kaggle.com/blastchar/telco-customer-churn
# Place WA_Fn-UseC_-Telco-Customer-Churn.csv in the data/ folder

# Run notebooks in order (01 → 04) to reproduce the model
# Then launch the dashboard
streamlit run app.py
```

---

## Business Recommendations

Based on model insights:

1. **Incentivize long-term contracts** — Converting month-to-month customers to annual contracts is the single highest-impact retention lever (reduces churn by ~15x)
2. **Onboard new high-paying customers aggressively** — Customers in their first 6 months paying above average are the highest flight risk
3. **Bundle OnlineSecurity and TechSupport** — Offer free 3-month trials to at-risk customers; these services cut churn from 40% to 15%
4. **Investigate Fiber Optic pricing** — 41.9% churn suggests customers don't feel they're getting value
5. **Migrate electronic check customers to auto-pay** — Offer a bill credit as an incentive

---

## Author

**Subramani Mokkala** — Aspiring Data Scientist

[LinkedIn](https://www.linkedin.com/in/subramani-mokkala/) | [GitHub](https://github.com/SubramaniMokkala)