# Customer Churn Prediction Application

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import joblib
import os

import os
st.write("Files in app directory:", os.listdir())

# Optional safety check
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

clf_weighted = joblib.load(os.path.join(BASE_DIR, "clf_weighted.pkl"))
clf_smote = joblib.load(os.path.join(BASE_DIR, "clf_smote.pkl"))
rf_clf = joblib.load(os.path.join(BASE_DIR, "rf_clf.pkl"))
gb_clf = joblib.load(os.path.join(BASE_DIR, "gb_clf.pkl"))

# --------------------------------------------------
# Page Configuration (MUST be first Streamlit command)
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# --------------------------------------------------
# Base directory
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# Load cleaned dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, "data", "telco_churn_cleaned.csv")
    return pd.read_csv(data_path)

df = load_data()

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "model", "final_churn_model.pkl")
    return joblib.load(model_path)

model = load_model()

# Features

EXPECTED_FEATURES = list(model.feature_names_in_)

def get_feature_groups(trained_model):
    preprocessor = trained_model.named_steps["preprocessor"]
    num, cat = [], []

    for name, _, cols in preprocessor.transformers_:
        if name == "num":
            num = list(cols)
        elif name == "cat":
            cat = list(cols)
    return num, cat

NUMERIC_FEATURES, CATEGORICAL_FEATURES = get_feature_groups(model)

# Feature Assembly
def assemble_features(user_input):
    data = {col: 0 for col in EXPECTED_FEATURES}

    data["Tenure Months"] = user_input["tenure"]
    data["Monthly Charges"] = user_input["monthly_charges"]
    data["Contract"] = user_input["contract"]
    data["Internet Service"] = user_input["internet_service"]
    data["Tech Support"] = user_input["tech_support"]
    data["Online Security"] = user_input["online_security"]
    data["Payment Method"] = user_input["payment_method"]

    tenure_safe = max(user_input["tenure"], 1)

    if "Charges per Tenure" in data:
        data["Charges per Tenure"] = user_input["monthly_charges"] / tenure_safe

    if "Early Tenure Flag" in data:
        data["Early Tenure Flag"] = 1 if user_input["tenure"] < 12 else 0

    neutral_defaults = {
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Paperless Billing": "Yes",
        "Country": "United States",
        "State": "California",
        "City": "Unknown",
        "Zip Code": 90001,
        "Latitude": 34.0,
        "Longitude": -118.0,
        "Count": 1
    }

    for col, val in neutral_defaults.items():
        if col in data:
            data[col] = val

    df_input = pd.DataFrame([data])

    for col in NUMERIC_FEATURES:
        if col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str)

    return df_input


# Sidebar Navigation
st.sidebar.title("Navigation")
if "page" not in st.session_state:
    st.session_state["page"] = "Overview"

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Data Insights",
        "Model Performance",
        "Churn Prediction",
        "Retention Recommendation"
    ],
    index=[
        "Overview",
        "Data Insights",
        "Model Performance",
        "Churn Prediction",
        "Retention Recommendation"
    ].index(st.session_state["page"])
)


# OVERVIEW
if page == "Overview":
    st.title("Customer Churn Prediction System")

    st.write(
        "This application demonstrates an end-to-end machine learning solution "
        "for predicting customer churn and supporting business-driven retention decisions."
    )

    st.divider()
    st.subheader("Explore the Application")

    # ROW 1
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(
    os.path.join(BASE_DIR, "images", "data.jpg"),
    use_container_width=True
)

        st.markdown("### Data Insights")
        st.write(
            "Explore customer behavior, churn patterns, "
            "and key business drivers through structured analysis."
        )
        if st.button("Go to Data Insights"):
            st.session_state["page"] = "Data Insights"

    with col2:
        st.image(
    os.path.join(BASE_DIR, "images", "model.png"),
    use_container_width=True
)
        st.markdown("### Model Performance")
        st.write(
            "Evaluate model reliability using business-focused metrics, "
            "error analysis, and feature importance."
        )
        if st.button("Go to Model Performance"):
            st.session_state["page"] = "Model Performance"

   

    # ROW 2
    col3, col4 = st.columns(2)

    with col3:
        st.image(
    os.path.join(BASE_DIR, "images", "churn.png"),
    use_container_width=True
)
        st.markdown("### Churn Prediction")
        st.write(
            "Predict churn probability for individual customers "
            "and receive risk-based retention recommendations."
        )
        if st.button("Go to Churn Prediction"):
            st.session_state["page"] = "Churn Prediction"

    with col4:
        st.image("images/retention.jpg", use_container_width=True)
        st.markdown("### Retention Recommendation")
        st.write(
            "Review the business decision framework that guides "
            "cost-effective customer retention strategies."
        )
        if st.button("Go to Retention Recommendation"):
            st.session_state["page"] = "Retention Recommendation"




# DATA INSIGHTS
elif page == "Data Insights":
    st.title("Data Insights: Understanding Customer Churn")

    st.write(
        "This section presents business-driven insights derived from the "
        "cleaned historical dataset used to train the churn prediction model. "
        "The goal is to understand why customers churn and which segments "
        "require targeted retention strategies."
    )

    st.divider()

    # 1. Target Understanding
    st.subheader("1. Churn Distribution (Target Understanding)")

    churn_dist = df["Churn Value"].value_counts(normalize=True) * 100
    churn_dist_df = churn_dist.rename({0: "Non-Churn", 1: "Churn"})

    st.write("Churn vs Non-Churn Percentage:")
    st.write(churn_dist_df)

    st.bar_chart(churn_dist_df)

    st.write(
        "The dataset shows a clear class imbalance, with fewer churned customers "
        "compared to retained customers. This confirms the need for imbalance-aware "
        "modeling and business-focused evaluation metrics."
    )

    st.divider()

    # 2. Tenure & Customer Lifecycle
    st.subheader("2. Tenure and Customer Lifecycle Analysis")

    tenure_summary = df.groupby("Churn Value")["Tenure Months"].mean()
    tenure_summary.index = ["Non-Churn", "Churn"]

    st.write("Average Tenure (Months):")
    st.write(tenure_summary)

    st.bar_chart(tenure_summary)

    st.write(
        "Churned customers typically have much lower tenure, indicating that "
        "early-stage customers are the most vulnerable to churn."
    )

    st.divider()

    # 3. Monetary & Customer Value Impact
    st.subheader("3. Monetary and Customer Value Impact")

    charges_summary = df.groupby("Churn Value")["Monthly Charges"].mean()
    cltv_summary = df.groupby("Churn Value")["CLTV"].mean()

    charges_summary.index = ["Non-Churn", "Churn"]
    cltv_summary.index = ["Non-Churn", "Churn"]

    st.write("Average Monthly Charges:")
    st.write(charges_summary)
    st.bar_chart(charges_summary)

    st.write("Average Customer Lifetime Value (CLTV):")
    st.write(cltv_summary)
    st.bar_chart(cltv_summary)

    st.write(
        "Churn among high-value customers leads to significant revenue loss, "
        "making selective and cost-aware retention strategies essential."
    )

    st.divider()

    # 4. Contract & Payment Behaviour
    st.subheader("4. Contract Type and Payment Behaviour")

    contract_churn = df.groupby("Contract")["Churn Value"].mean().sort_values(ascending=False)
    payment_churn = df.groupby("Payment Method")["Churn Value"].mean().sort_values(ascending=False)

    st.write("Churn Rate by Contract Type:")
    st.write(contract_churn)
    st.bar_chart(contract_churn)

    st.write("Churn Rate by Payment Method:")
    st.write(payment_churn)
    st.bar_chart(payment_churn)

    st.write(
        "Customers on month-to-month contracts and electronic check payments "
        "show higher churn rates, indicating lower long-term commitment."
    )

    st.divider()

    
    # 5. Service & Usage Behaviour
    st.subheader("5. Service and Usage Behaviour")

    service_cols = [
        "Internet Service",
        "Online Security",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies"
    ]

    for col in service_cols:
        if col in df.columns:
            service_churn = df.groupby(col)["Churn Value"].mean().sort_values(ascending=False)
            st.write(f"Churn Rate by {col}:")
            st.write(service_churn)
            st.bar_chart(service_churn)

    st.write(
        "Customers lacking value-added services such as security and technical support "
        "tend to churn more, suggesting dissatisfaction or unmet expectations."
    )

    st.divider()

     
    # 6. Customer Demographics
    st.subheader("6. Customer Demographics")

    demo_cols = ["Senior Citizen", "Partner", "Dependents"]

    for col in demo_cols:
        if col in df.columns:
            demo_churn = df.groupby(col)["Churn Value"].mean()
            st.write(f"Churn Rate by {col}:")
            st.write(demo_churn)
            st.bar_chart(demo_churn)

    st.write(
        "Certain demographic segments, such as senior citizens and customers without "
        "dependents, show different churn behavior patterns."
    )

    st.divider()

    # 7. Engagement & Dissatisfaction Signals
    st.subheader("7. Engagement and Dissatisfaction Indicators")

    engagement_cols = [
        "Engagement Score",
        "Low Engagement Flag",
        "Dissatisfaction Score"
    ]

    for col in engagement_cols:
        if col in df.columns:
            st.write(f"{col} vs Churn:")
            st.write(df.groupby("Churn Value")[col].mean())

    st.write(
        "Lower engagement and higher dissatisfaction scores are strong early indicators "
        "of churn and can be used for proactive intervention."
    )

# MODEL PERFORMANCE
elif page == "Model Performance":

    st.title("Model Performance Evaluation")

    from sklearn.metrics import (
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_curve
    )
    import matplotlib.pyplot as plt
    import shap

    X = df.drop(columns=["Churn Value"])
    y = df["Churn Value"]

    THRESHOLD = 0.55

    # =========================
    # Helper function for metrics
    # =========================
    def compute_metrics(model, X, y):
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= THRESHOLD).astype(int)

        return pd.DataFrame({
            "Metric": ["ROC-AUC", "Precision", "Recall", "F1-Score"],
            "Value": [
                roc_auc_score(y, y_prob),
                precision_score(y, y_pred),
                recall_score(y, y_pred),
                f1_score(y, y_pred)
            ]
        })

    # =========================
    # 1️⃣ Logistic Regression (Class Weighted)
    # =========================
    st.subheader("1. Logistic Regression (Class Weighted)")
    st.dataframe(
        compute_metrics(clf_weighted, X, y)
        .style.format({"Value": "{:.3f}"})
    )

    # =========================
    # 2️⃣ Logistic Regression + SMOTE
    # =========================
    st.subheader("2. Logistic Regression with SMOTE")
    st.dataframe(
        compute_metrics(clf_smote, X, y)
        .style.format({"Value": "{:.3f}"})
    )

    # =========================
    # 3️⃣ Random Forest (Class Weighted)
    # =========================
    st.subheader("3. Random Forest")
    st.dataframe(
        compute_metrics(rf_clf, X, y)
        .style.format({"Value": "{:.3f}"})
    )

    # =========================
    # 4️⃣ Gradient Boosting
    # =========================
    st.subheader("4. Gradient Boosting")
    st.dataframe(
        compute_metrics(gb_clf, X, y)
        .style.format({"Value": "{:.3f}"})
    )

    # =========================
    # 5️⃣ Confusion Matrix (Random Forest)
    # =========================
    st.subheader("5. Confusion Matrix")

    y_prob_rf = rf_clf.predict_proba(X)[:, 1]
    y_pred_rf = (y_prob_rf >= THRESHOLD).astype(int)

    cm = confusion_matrix(y, y_pred_rf)

    fig, ax = plt.subplots(figsize=(3,2))
    ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix – Random Forest")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    st.pyplot(fig, use_container_width = False)

    # =========================
    # 6️⃣ ROC Curve & ROC-AUC (Random Forest)
    # =========================
    st.subheader("6. ROC Curve")

    fpr, tpr, _ = roc_curve(y, y_prob_rf)
    auc_score = roc_auc_score(y, y_prob_rf)

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Random Forest")
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig, use_container_width = False)

# CHURN PREDICTION
elif page == "Churn Prediction":

    st.title("Customer Churn Prediction")

    # =========================
    # INPUT SECTION
    # =========================
    st.subheader("Customer Profile Inputs")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=120,
            value=12
        )

        monthly_charges = st.slider(
            "Monthly Charges",
            min_value=0.0,
            max_value=500.0,
            value=70.0
        )

        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["Fiber optic", "DSL", "No"]
        )

    with col2:
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    # =========================
    # PREDICTION
    # =========================
    if st.button("Predict Churn Risk"):

        user_input = {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "contract": contract,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "online_security": online_security,
            "payment_method": payment_method
        }

        model_input = assemble_features(user_input)
        churn_probability = model.predict_proba(model_input)[0][1]
        churn_percentage = churn_probability * 100

        # =========================
        # RISK LEVEL LOGIC (UNCHANGED)
        # =========================
        if churn_probability >= 0.55:
            risk_level = "High Risk"
        elif churn_probability >= 0.35:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        # =========================
        # CUSTOMER VALUE LOGIC (UNCHANGED)
        # =========================
        if tenure == 0:
            customer_value = "Low Value"
        elif tenure >= 12 and monthly_charges >= 60:
            customer_value = "High Value"
        elif tenure >= 6 or monthly_charges >= 40:
            customer_value = "Medium Value"
        else:
            customer_value = "Low Value"

        # =========================
        # SPEEDOMETER / GAUGE
        # =========================
        import plotly.graph_objects as go

        st.subheader("Churn Risk Indicator")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_percentage,
            title={"text": "Churn Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred"},
                "steps": [
                    {"range": [0, 35], "color": "#4CAF50"},
                    {"range": [35, 55], "color": "#FFC107"},
                    {"range": [55, 100], "color": "#F44336"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": churn_percentage,
                },
            },
        ))

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # TEXT OUTPUT (UNCHANGED)
        # =========================
        st.subheader("Prediction Result")
        st.write(f"Churn Probability: {churn_probability:.2%}")
        st.write(f"Risk Category: {risk_level}")

        st.subheader("Customer Assessment")
        st.write(f"Customer Value Segment: {customer_value}")

        # =========================
        # RETENTION DECISION (100% UNCHANGED)
        # =========================
        st.subheader("Retention Decision")

        if risk_level == "High Risk" and customer_value == "High Value":
            st.write("High-value customer at high churn risk.")
            st.write("• Recommend proactive outreach with personalized retention offers.")
            st.write("• Analyze customer complaints related to services such as:")
            st.write("  - Poor network quality")
            st.write("  - Delays in issue resolution by support teams")
            st.write("  - Increasing costs over time (monthly or yearly)")

        elif risk_level == "High Risk" and customer_value == "Medium Value":
            st.write("Customer has high churn risk with moderate value.")
            st.write("• Consider low-cost retention offers.")
            st.write("• Monitor service quality and billing experience.")
            st.write("• Analyze complaints related to our services")

        elif risk_level == "High Risk" and customer_value == "Low Value":
            st.write("Customer has high churn risk but low lifetime value.")
            st.write("• Retention is not cost-effective.")
            st.write("• No proactive retention action recommended.")

        elif risk_level == "Medium Risk" and customer_value == "High Value":
            st.write("High-value customer with moderate churn risk.")
            st.write("• Monitor closely and provide engagement offers.")

        elif risk_level == "Medium Risk":
            st.write("Moderate churn risk detected.")
            st.write("• Continue monitoring without immediate intervention.")

        else:
            st.write("Low churn risk detected.")
            st.write("• No retention action required.")

# RETENTION RECOMMENDATION

elif page == "Retention Recommendation":
    st.title("Retention Recommendation Framework")

    st.write(
        "This section presents a business-driven retention decision framework. "
        "It explains how churn risk and customer value are combined to determine "
        "whether retention actions are cost-effective."
    )

    st.divider()

    st.subheader("Retention Strategy Matrix")

    retention_table = pd.DataFrame({
        "Churn Risk Level": [
            "High Risk",
            "High Risk",
            "High Risk",
            "Medium Risk",
            "Medium Risk",
            "Medium Risk",
            "Low Risk"
        ],
        "Customer Value Segment": [
            "High Value",
            "Medium Value",
            "Low Value",
            "High Value",
            "Medium Value",
            "Low Value",
            "Any"
        ],
        "Business Interpretation": [
            "High revenue customer is very likely to churn",
            "Moderate revenue customer is very likely to churn",
            "Low revenue customer is very likely to churn",
            "High revenue customer shows early churn signals",
            "Average customer shows moderate churn risk",
            "Low revenue customer shows moderate churn risk",
            "Customer is stable with low churn probability"
        ],
        "Recommended Retention Action": [
            "Immediate proactive retention with personalized offers and senior support",
            "Targeted low-cost offers and service quality review",
            "No proactive retention (cost not justified)",
            "Close monitoring with engagement offers and service improvement",
            "Monitor behavior and send soft engagement messages",
            "No immediate action; observe long-term behavior",
            "No retention required; maintain service quality"
        ],
        "Business Reasoning": [
            "Losing a high-value customer causes significant revenue loss",
            "Retention may be profitable if cost is controlled",
            "Retention cost exceeds expected future revenue",
            "Early intervention can prevent future high-value churn",
            "Retention benefit is uncertain; avoid aggressive offers",
            "Customer lifetime value is low",
            "Retention investment not required"
        ]
    })

    st.dataframe(retention_table, use_container_width=True)

    

    st.write(
        "This framework is independent of the model prediction output and "
        "serves as a decision guideline for business teams. "
        "Actual retention actions are applied after evaluating both churn risk "
        "and customer value."
    )







