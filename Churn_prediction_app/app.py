import streamlit as st
import pandas as pd
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")

# ---------------------------
# CUSTOM CSS
# ---------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.block-container {
    padding-top: 0.5rem;
}

/* KPI CARD */
.kpi-card {
    background-color: #1c1f26;
    padding: 10px;
    border-radius: 12px;
    border: 1px solid #333;
    text-align: left;
}

/* TOP ROW (TITLE + ARROW) */
.kpi-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* TITLE */
.kpi-title {
    font-size: 11px;
    color: #9aa0a6;
}

/* VALUE */
.kpi-value {
    font-size: 20px;
    font-weight: bold;
    color: white;
    margin-top: 5px;
}

/* SMALL TEXT */
.kpi-sub {
    font-size: 9px;
    color: #6c757d;
    margin-top: 4px;
}

/* ARROWS */
.up {
    color: #00c853;
    font-size: 14px;
}

.down {
    color: #ff5252;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD DATA
# ---------------------------

@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "data", "telco_churn_cleaned.csv")
    return pd.read_csv(data_path)

df = load_data()

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------------------
# CREATE CURRENT vs PREVIOUS SPLIT
# ---------------------------
split_index = int(len(df) * 0.5)

prev_df = df.iloc[:split_index]
curr_df = df.iloc[split_index:]

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "model", "final_churn_model.pkl")
    return joblib.load(model_path)

model = load_model()

# ---------------------------
# MAIN TITLE (LEFT)
# ---------------------------
st.markdown("""
<h3 style='text-align: left; color: white; margin-top: 20px;'>
Customer Churn Prediction & Retention System
</h3>
""", unsafe_allow_html=True)

# ---------------------------
# KPI CALCULATIONS
# ---------------------------
# ---------------------------
# KPI CALCULATIONS (REAL)
# ---------------------------

# CURRENT VALUES
total_customers = len(curr_df)
churn_rate = curr_df["Churn Value"].mean() * 100
avg_tenure = curr_df["Tenure Months"].mean()
avg_charges = curr_df["Monthly Charges"].mean()
high_risk_customers = curr_df[curr_df["Churn Value"] == 1].shape[0]
revenue_at_risk = curr_df[curr_df["Churn Value"] == 1]["Monthly Charges"].sum()

# PREVIOUS VALUES
prev_total = len(prev_df)
prev_churn = prev_df["Churn Value"].mean() * 100
prev_tenure = prev_df["Tenure Months"].mean()
prev_charges = prev_df["Monthly Charges"].mean()
prev_high_risk = prev_df[prev_df["Churn Value"] == 1].shape[0]
prev_revenue = prev_df[prev_df["Churn Value"] == 1]["Monthly Charges"].sum()

def calc_change(curr, prev):
    if prev == 0:
        return 0
    return ((curr - prev) / prev) * 100

chg_total = calc_change(total_customers, prev_total)
chg_churn = calc_change(churn_rate, prev_churn)
chg_tenure = calc_change(avg_tenure, prev_tenure)
chg_charges = calc_change(avg_charges, prev_charges)
chg_risk = calc_change(high_risk_customers, prev_high_risk)
chg_revenue = calc_change(revenue_at_risk, prev_revenue)
# ---------------------------
# KPI ROW
# ---------------------------
col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.markdown(f"""
<div class="kpi-card">
    <div class="kpi-top">
        <div class="kpi-title">Total Customers</div>
        <div class="{'up' if chg_total >= 0 else 'down'}">
{'▲' if chg_total >= 0 else '▼'} {abs(chg_total):.1f}%
</div>
    </div>
    <div class="kpi-value">{total_customers}</div>
    <div class="kpi-sub">vs previous period</div>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="kpi-card">
    <div class="kpi-top">
        <div class="kpi-title">Churn Rate</div>
        <div class="{'up' if chg_churn >= 0 else 'down'}">
{'▲' if chg_churn >= 0 else '▼'} {abs(chg_churn):.1f}%
</div>
    </div>
    <div class="kpi-value">{churn_rate:.2f}%</div>
    <div class="kpi-sub">vs previous period</div>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="kpi-card">
    <div class="kpi-top">
        <div class="kpi-title">Avg Tenure</div>
        <div class="{'up' if chg_tenure >= 0 else 'down'}">
{'▲' if chg_tenure >= 0 else '▼'} {abs(chg_tenure):.1f}%
</div>
    </div>
    <div class="kpi-value">{avg_tenure:.1f}</div>
    <div class="kpi-sub">vs previous period</div>
</div>
""", unsafe_allow_html=True)

col4.markdown(f"""
<div class="kpi-card">
    <div class="kpi-top">
        <div class="kpi-title">Avg Charges</div>
        <div class="{'up' if chg_charges >= 0 else 'down'}">
{'▲' if chg_charges >= 0 else '▼'} {abs(chg_charges):.1f}%
</div>
    </div>
    <div class="kpi-value">${avg_charges:.2f}</div>
    <div class="kpi-sub">vs previous period</div>
</div>
""", unsafe_allow_html=True)

col5.markdown(f"""
<div class="kpi-card">
    <div class="kpi-top">
        <div class="kpi-title">High Risk Customers</div>
        <div class="{'up' if chg_risk >= 0 else 'down'}">
{'▲' if chg_risk >= 0 else '▼'} {abs(chg_risk):.1f}%
</div>
    </div>
    <div class="kpi-value">{high_risk_customers}</div>
    <div class="kpi-sub">vs previous period</div>
</div>
""", unsafe_allow_html=True)

col6.markdown(f"""
<div class="kpi-card">
    <div class="kpi-top">
        <div class="kpi-title">Revenue at Risk</div>
        <div class="{'up' if chg_revenue >= 0 else 'down'}">
{'▲' if chg_revenue >= 0 else '▼'} {abs(chg_revenue):.1f}%
</div>
    </div>
    <div class="kpi-value">${revenue_at_risk:,.0f}</div>
    <div class="kpi-sub">vs previous period</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# INFO CARDS (4 BOXES)
# ---------------------------

# CSS for small cards
st.markdown("""
<style>
.info-card {
    background-color: #1c1f26;
    padding: 10px;
    border-radius: 10px;
    border: 1px solid #333;
    margin-bottom: 5px;
}

.info-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: white;
    font-size: 13px;
}

.arrow {
    color: #9aa0a6;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# 4 columns
c1, c2, c3, c4 = st.columns(4)

# ---------------------------
# BOX 1: ML MODELS
# ---------------------------
with c1:
    with st.expander("ML Models Used "):
        st.markdown("""
        - Logistic Regression  
        - Random Forest  
        - Gradient Boosting  
        """)

# ---------------------------
# BOX 2: CUSTOMER SEGMENTS
# ---------------------------
with c2:
    with st.expander("Customer Segments "):
        st.markdown("""
        - Senior Citizens → Higher churn  
        - Low Tenure Customers → High churn  
        - High Monthly Charges → High churn  
        """)

# ---------------------------
# BOX 3: CHURN DRIVERS
# ---------------------------
with c3:
    with st.expander("Churn Drivers "):
        st.markdown("""
        - Month-to-Month Contracts → High churn  
        - Short Tenure → High risk  
        - Expensive Plans → More churn  
        """)

# ---------------------------
# BOX 4: RETENTION STRATEGY
# ---------------------------
with c4:
    with st.expander("Retention Strategy "):
        st.markdown("""
        - Offer discounts to high-risk users  
        - Encourage long-term contracts  
        - Target senior citizens with plans  
        """)



# ---------------------------
# HELPER: GET FEATURE IMPORTANCE FROM PIPELINE
# ---------------------------
def get_feature_importance(model, X):
    try:
        # if pipeline → get last step
        if hasattr(model, "named_steps"):
            last_model = list(model.named_steps.values())[-1]
        else:
            last_model = model

        if hasattr(last_model, "feature_importances_"):
            return pd.Series(last_model.feature_importances_, index=X.columns)
        else:
            return None
    except:
        return None


# ---------------------------
# CHART ROW 1
# ---------------------------
r1c1, r1c2, r1c3 = st.columns(3)

# 1. CHURN PIE
with r1c1:
    churn_counts = curr_df["Churn Value"].value_counts()

    fig1 = go.Figure(data=[go.Pie(
        labels=["Retained", "Churned"],
        values=[churn_counts[0], churn_counts[1]],
        hole=0.7,
        marker=dict(colors=["#1f77b4", "#ff7f0e"]),
        textinfo='label+value',
        textfont=dict(size=12, color="white")
    )])

    churn_percent = (churn_counts[1] / churn_counts.sum()) * 100

    fig1.update_layout(
        title="Churn",
        height=350,

        margin=dict(t=50, b=0, l=50, r=100),

        annotations=[dict(
            text=f"{churn_percent:.1f}%",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=18, color="white")
        )],
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white"),
        legend=dict(orientation="h", x=0.18, y=-0.1)
    )

    st.plotly_chart(fig1, use_container_width=True)


# 2. TENURE HIST
with r1c2:
    fig2 = px.histogram(
        curr_df,
        x="Tenure Months",
        color="Churn Value",
        nbins=20,
        barmode="overlay",
        
        color_discrete_map={0:"#1f77b4", 1:"#ff7f0e"}
    )

    fig2.update_layout(
        title="Tenure",
        height=360,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white")
    )

    st.plotly_chart(fig2, use_container_width=True)


# 3. CHARGES BOX
with r1c3:
    fig3 = px.box(
        curr_df,
        x="Churn Value",
        y="Monthly Charges",
        color="Churn Value",
        color_discrete_map={0:"#1f77b4", 1:"#ff7f0e"}
    )

    fig3.update_layout(
        title="Charges",
        height=355,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white"),
        showlegend=False
    )

    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------
# CHART ROW 2
# ---------------------------
r2c1, r2c2, r2c3, r2c4 = st.columns(4)

# 4. CONTRACT
with r2c1:
    contract = curr_df.groupby("Contract")["Churn Value"].mean().reset_index()

    fig4 = px.bar(contract, x="Contract", y="Churn Value",
                  color_discrete_sequence=["#ff7f0e"])

    fig4.update_layout(
        title="Contract",
        height=350,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white")
    )

    st.plotly_chart(fig4, use_container_width=True)


# 5. PAYMENT
with r2c2:
    pay = curr_df.groupby("Payment Method")["Churn Value"].mean().reset_index()

    fig5 = px.bar(pay, x="Payment Method", y="Churn Value",
                  color_discrete_sequence=["#ff7f0e"])

    fig5.update_layout(
        title="Payment",
        height=350,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white")
    )

    st.plotly_chart(fig5, use_container_width=True)


# 6. PAYMENT SPLIT
with r2c3:
    pay_split = curr_df.groupby(["Payment Method", "Churn Value"]).size().reset_index(name="count")

    fig6 = px.bar(
        pay_split,
        x="Payment Method",
        y="count",
        color="Churn Value",
        barmode="group",
        color_discrete_map={0:"#1f77b4", 1:"#ff7f0e"}
    )

    fig6.update_layout(
        title="Pay Split",
        height=350,
        width= 400,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white")
    )

    st.plotly_chart(fig6, use_container_width=True)


# 7. FEATURE IMPORTANCE (FIXED)
with r2c4:
    X = curr_df.drop(columns=["Churn Value"])

    feat_imp = get_feature_importance(model, X)

    if feat_imp is not None:
        feat_imp = feat_imp.sort_values(ascending=False).head(5)

        fig7 = px.bar(
            feat_imp,
            x=feat_imp.values,
            y=feat_imp.index,
            orientation='h',
            color_discrete_sequence=["#1f77b4"]
        )
    else:
        fig7 = go.Figure()
        fig7.add_annotation(text="No Feature Importance", showarrow=False, font=dict(color="white"))

    fig7.update_layout(
        title="Features",
        height=350,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white")
    )

    st.plotly_chart(fig7, use_container_width=True)


# ---------------------------
# CHART ROW 3
# ---------------------------
r3c1, r3c2, r3c3 = st.columns(3)

# 8. SERVICE
with r3c1:
    service = curr_df.groupby(["Tech Support", "Churn Value"]).size().reset_index(name="count")

    fig8 = px.bar(
        service,
        x="Tech Support",
        y="count",
        color="Churn Value",
        barmode="group",
        color_discrete_map={0:"#1f77b4", 1:"#ff7f0e"}
    )

    fig8.update_layout(
        title="Service",
        height=350,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white")
    )

    st.plotly_chart(fig8, use_container_width=True)


with r3c2:

    # REAL METRICS
    X = curr_df.drop(columns=["Churn Value"])
    y = curr_df["Churn Value"]

    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    metrics = ["Acc", "Prec", "Rec", "F1"]
    values = [acc, prec, rec, f1]

    # FIND BEST METRIC
    max_val = max(values)

    # COLORS (highlight best)
    colors = ["#1f77b4" if v != max_val else "#00c853" for v in values]

    fig9 = go.Figure()

    fig9.add_bar(
        x=metrics,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],   # 🔥 value labels
        textposition='outside'
    )

    fig9.update_layout(
        title="Model",
        height=350,
        yaxis=dict(range=[0,1]),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="white"),
        margin=dict(t=30, b=20, l=20, r=20)
    )

    st.plotly_chart(fig9, use_container_width=True)


# 10. GAUGE
with r3c3:
    prob = model.predict_proba(curr_df.drop(columns=["Churn Value"]))[:,1].mean()

    fig10 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Churn %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "red"}
            ]
        }
    ))

    fig10.update_layout(
        height=350,
        paper_bgcolor="#0E1117",
        font=dict(color="white")
    )

    st.plotly_chart(fig10, use_container_width=True)
