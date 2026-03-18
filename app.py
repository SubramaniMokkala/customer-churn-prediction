import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 8px 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Risk badge */
    .risk-high {
        background-color: #3d1515;
        border: 1px solid #e74c3c;
        border-radius: 8px;
        padding: 12px 20px;
        color: #e74c3c;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
    }
    
    .risk-medium {
        background-color: #2d2100;
        border: 1px solid #f39c12;
        border-radius: 8px;
        padding: 12px 20px;
        color: #f39c12;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
    }
    
    .risk-low {
        background-color: #0d2818;
        border: 1px solid #2ecc71;
        border-radius: 8px;
        padding: 12px 20px;
        color: #2ecc71;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
    }
    
    /* Progress bar container */
    .progress-container {
        background-color: #21262d;
        border-radius: 10px;
        height: 12px;
        width: 100%;
        margin: 8px 0;
    }
    
    .progress-bar {
        height: 12px;
        border-radius: 10px;
        transition: width 0.3s ease;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #58a6ff;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }

    /* Divider */
    hr {
        border-color: #30363d;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('models/churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Header
st.markdown("# Customer Churn Prediction System")
st.markdown("<p style='color:#8b949e'>Identify at-risk customers and enable proactive retention</p>", 
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.markdown("## Customer Profile")
st.sidebar.markdown("---")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
contract = st.sidebar.selectbox("Contract Type",
                                 ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service",
                                         ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox("Payment Method",
                                       ["Electronic check", "Mailed check",
                                        "Bank transfer (automatic)",
                                        "Credit card (automatic)"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Additional Services**")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    online_security = st.checkbox("Security")
    device_protection = st.checkbox("Device")
    streaming_tv = st.checkbox("Stream TV")
    phone_service = st.checkbox("Phone", value=True)
with col_b:
    online_backup = st.checkbox("Backup")
    tech_support = st.checkbox("Support")
    streaming_movies = st.checkbox("Movies")
    multiple_lines = st.checkbox("Multi Lines")

st.sidebar.markdown("---")
st.sidebar.markdown("**Demographics**")
paperless_billing = st.sidebar.checkbox("Paperless Billing")
senior_citizen = st.sidebar.checkbox("Senior Citizen")
partner = st.sidebar.checkbox("Has Partner")
dependents = st.sidebar.checkbox("Has Dependents")

def prepare_features(tenure, monthly_charges, contract, internet_service,
                     payment_method, online_security, online_backup,
                     device_protection, tech_support, streaming_tv,
                     streaming_movies, phone_service, multiple_lines,
                     paperless_billing, senior_citizen, partner, dependents):

    charges_to_tenure_ratio = monthly_charges / (tenure + 1)
    services = [online_security, online_backup, device_protection,
                tech_support, streaming_tv, streaming_movies]
    service_adoption_score = sum(services)
    high_risk_combo = int(contract == "Month-to-month" and monthly_charges > 65)

    if tenure <= 12:
        tenure_group = "New"
    elif tenure <= 24:
        tenure_group = "Developing"
    elif tenure <= 48:
        tenure_group = "Established"
    else:
        tenure_group = "Loyal"

    features = {
        'SeniorCitizen': int(senior_citizen),
        'Partner': int(partner),
        'Dependents': int(dependents),
        'tenure': tenure,
        'PhoneService': int(phone_service),
        'PaperlessBilling': int(paperless_billing),
        'MonthlyCharges': monthly_charges,
        'charges_to_tenure_ratio': charges_to_tenure_ratio,
        'service_adoption_score': service_adoption_score,
        'high_risk_combo': high_risk_combo,
        'MultipleLines_No phone service': int(not phone_service),
        'MultipleLines_Yes': int(multiple_lines),
        'InternetService_Fiber optic': int(internet_service == "Fiber optic"),
        'InternetService_No': int(internet_service == "No"),
        'OnlineSecurity_No internet service': int(internet_service == "No"),
        'OnlineSecurity_Yes': int(online_security),
        'OnlineBackup_No internet service': int(internet_service == "No"),
        'OnlineBackup_Yes': int(online_backup),
        'DeviceProtection_No internet service': int(internet_service == "No"),
        'DeviceProtection_Yes': int(device_protection),
        'TechSupport_No internet service': int(internet_service == "No"),
        'TechSupport_Yes': int(tech_support),
        'StreamingTV_No internet service': int(internet_service == "No"),
        'StreamingTV_Yes': int(streaming_tv),
        'StreamingMovies_No internet service': int(internet_service == "No"),
        'StreamingMovies_Yes': int(streaming_movies),
        'Contract_One year': int(contract == "One year"),
        'Contract_Two year': int(contract == "Two year"),
        'PaymentMethod_Bank transfer (automatic)': int(payment_method == "Bank transfer (automatic)"),
        'PaymentMethod_Credit card (automatic)': int(payment_method == "Credit card (automatic)"),
        'PaymentMethod_Electronic check': int(payment_method == "Electronic check"),
        'tenure_group_Developing': int(tenure_group == "Developing"),
        'tenure_group_Established': int(tenure_group == "Established"),
        'tenure_group_Loyal': int(tenure_group == "Loyal"),
    }

    df = pd.DataFrame([features])
    numerical_cols = ['tenure', 'MonthlyCharges',
                      'charges_to_tenure_ratio', 'service_adoption_score']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    return df


input_df = prepare_features(
    tenure, monthly_charges, contract, internet_service,
    payment_method, online_security, online_backup,
    device_protection, tech_support, streaming_tv,
    streaming_movies, phone_service, multiple_lines,
    paperless_billing, senior_citizen, partner, dependents
)

churn_prob = model.predict_proba(input_df)[0][1]
risk_pct = churn_prob * 100

if churn_prob >= 0.7:
    risk_label = "HIGH RISK"
    risk_class = "risk-high"
    bar_color = "#e74c3c"
elif churn_prob >= 0.3:
    risk_label = "MEDIUM RISK"
    risk_class = "risk-medium"
    bar_color = "#f39c12"
else:
    risk_label = "LOW RISK"
    risk_class = "risk-low"
    bar_color = "#2ecc71"

# Risk Score Section
st.markdown("<div class='section-header'>Risk Assessment</div>", 
            unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Churn Probability</div>
        <div class='metric-value' style='color:{bar_color}'>{risk_pct:.1f}%</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Tenure</div>
        <div class='metric-value' style='color:#58a6ff'>{tenure}mo</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Monthly Charges</div>
        <div class='metric-value' style='color:#58a6ff'>${monthly_charges}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Services Used</div>
        <div class='metric-value' style='color:#58a6ff'>
            {sum([online_security, online_backup, device_protection,
                  tech_support, streaming_tv, streaming_movies])}
        </div>
    </div>""", unsafe_allow_html=True)

# Risk badge & progress bar
st.markdown("<br>", unsafe_allow_html=True)
col5, col6 = st.columns([1, 2])

with col5:
    st.markdown(f"<div class='{risk_class}'>{risk_label}</div>",
                unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div style='margin-top:8px'>
        <div style='color:#8b949e; font-size:0.85rem; margin-bottom:4px'>
            Risk Level: {risk_pct:.1f}%
        </div>
        <div class='progress-container'>
            <div class='progress-bar' 
                 style='width:{risk_pct}%; background-color:{bar_color}'></div>
        </div>
        <div style='display:flex; justify-content:space-between; 
                    color:#8b949e; font-size:0.75rem; margin-top:4px'>
            <span>Low Risk</span>
            <span>Medium Risk</span>
            <span>High Risk</span>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# Analytics Section
st.markdown("<div class='section-header'>Model Analytics</div>",
            unsafe_allow_html=True)

col7, col8 = st.columns(2)

with col7:
    feature_importance = pd.DataFrame({
        'Feature': ['Charges/Tenure Ratio', 'Two Year Contract',
                    'High Risk Combo', 'Electronic Check',
                    'Monthly Charges', 'Fiber Optic',
                    'Online Security', 'Tech Support'],
        'Importance': [0.67, 0.34, 0.24, 0.20, 0.19, 0.15, 0.10, 0.09]
    }).sort_values('Importance')

    fig_imp = px.bar(feature_importance, 
                     x='Importance', y='Feature',
                     orientation='h',
                     title='Top Churn Predictors',
                     color='Importance',
                     color_continuous_scale='Blues')
    fig_imp.update_layout(
        paper_bgcolor='#161b22',
        plot_bgcolor='#161b22',
        font_color='#ffffff',
        title_font_color='#58a6ff'
    )
    st.plotly_chart(fig_imp, use_container_width=True)

with col8:
    st.markdown("<div style='margin-top:20px'>", unsafe_allow_html=True)

    business_metrics = [
        ("Annual Revenue Saved", "$637,850", "#2ecc71"),
        ("Campaign ROI", "468%", "#2ecc71"),
        ("Customers Saved / Year", "430", "#58a6ff"),
        ("ROC-AUC Score", "0.8495", "#58a6ff"),
        ("Churners Identified", "76.7%", "#f39c12"),
    ]

    for label, value, color in business_metrics:
        st.markdown(f"""
        <div class='metric-card' style='display:flex; 
             justify-content:space-between; align-items:center;
             padding:14px 20px; margin:6px 0'>
            <span style='color:#8b949e; font-size:0.9rem'>{label}</span>
            <span style='color:{color}; font-weight:700; 
                         font-size:1.1rem'>{value}</span>
        </div>""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#8b949e; font-size:0.85rem; padding:10px'>
    Built by Subramani Mokkala | XGBoost + SHAP + Streamlit | 
    <a href='https://github.com/SubramaniMokkala/customer-churn-prediction' 
       style='color:#58a6ff'>GitHub</a>
</div>""", unsafe_allow_html=True)