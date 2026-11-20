from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/model.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction — MLOps",
    page_icon="📊",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"Champion model not found at `{MODEL_PATH}`. "
            "Please run `dvc repro` or `dvc pull` first."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("📊 Customer Churn Prediction System")
st.caption(
    "Powered by a 5-stage DVC + Optuna + MLflow Multi-Criteria Champion Pipeline "
    "trained on 440,832 customer records."
)

with st.form("churn_prediction_form"):
    st.subheader("Customer Profile & Usage Inputs")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=35, step=1)
        gender = st.selectbox("Gender", options=["Female", "Male"])
        tenure = st.number_input("Tenure (months)", min_value=1, max_value=60, value=24, step=1)
        usage_freq = st.number_input("Usage Frequency (per month)", min_value=1, max_value=30, value=14, step=1)
        support_calls = st.number_input("Support Calls", min_value=0, max_value=15, value=2, step=1)

    with col2:
        payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=30, value=5, step=1)
        subscription_type = st.selectbox("Subscription Type", options=["Standard", "Basic", "Premium"])
        contract_length = st.selectbox("Contract Length", options=["Annual", "Quarterly", "Monthly"])
        total_spend = st.number_input("Total Spend ($)", min_value=50.0, max_value=5000.0, value=650.0, step=10.0)
        last_interaction = st.number_input("Last Interaction (days ago)", min_value=1, max_value=30, value=10, step=1)

    submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

if submitted:
    input_df = pd.DataFrame(
        [
            {
                "Age": float(age),
                "Tenure": float(tenure),
                "Usage Frequency": float(usage_freq),
                "Support Calls": float(support_calls),
                "Payment Delay": float(payment_delay),
                "Total Spend": float(total_spend),
                "Last Interaction": float(last_interaction),
                "Gender": gender,
                "Subscription Type": subscription_type,
                "Contract Length": contract_length,
            }
        ]
    )

    prediction = int(model.predict(input_df)[0])

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ Prediction: HIGH CHURN RISK (Customer Likely to Churn)")
    else:
        st.success("### ✅ Prediction: LOW CHURN RISK (Customer Likely to Stay)")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        proba_df = pd.DataFrame(
            {
                "Outcome": ["Retained (0)", "Churned (1)"],
                "Probability": [float(proba[0]), float(proba[1])],
            }
        ).set_index("Outcome")

        m1, m2 = st.columns(2)
        m1.metric("Retention Probability", f"{proba[0] * 100:.2f}%")
        m2.metric("Churn Probability", f"{proba[1] * 100:.2f}%")
        st.bar_chart(proba_df)
