import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Load Data and Model
# --------------------------------------------------

df = pd.read_csv("Customer Churn.csv")
model = joblib.load("customer_churn_model.pkl")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Customer Churn System")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Churn Analysis",
        "Model Performance",
        "New Customer Prediction"
    ]
)

# ==================================================
# 1. OVERVIEW
# ==================================================

if page == "Overview":

    st.title("📊 Customer Churn Prediction")
    st.write("XGBoost-based Customer Churn Analysis Dashboard")

    total_customers = len(df)
    churned_customers = df["Churn"].sum()
    churn_rate = churned_customers / total_customers * 100
    avg_customer_value = df["Customer Value"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", total_customers)
    col2.metric("Churned Customers", int(churned_customers))
    col3.metric("Churn Rate", f"{churn_rate:.2f}%")
    col4.metric("Avg Customer Value", f"{avg_customer_value:.2f}")

    st.subheader("Churn Distribution")

    churn_counts = df["Churn"].value_counts().rename(
        index={0: "No Churn", 1: "Churn"}
    )

    st.bar_chart(churn_counts)


# ==================================================
# 2. CHURN ANALYSIS
# ==================================================

elif page == "Churn Analysis":

    st.title("📈 Churn Analysis")

    st.subheader("Churn by Complaints")

    complaint_churn = pd.crosstab(
        df["Complains"],
        df["Churn"]
    )

    complaint_churn.columns = ["No Churn", "Churn"]

    st.bar_chart(complaint_churn)

    st.subheader("Average Customer Value by Churn")

    value_churn = df.groupby("Churn")["Customer Value"].mean()

    value_churn.index = ["No Churn", "Churn"]

    st.bar_chart(value_churn)

    st.subheader("Customer Value Distribution")

    st.write(
        "Comparison of customer value between churned and retained customers."
    )

    st.dataframe(
        df.groupby("Churn")["Customer Value"]
        .agg(["mean", "median", "min", "max"])
        .rename(index={0: "No Churn", 1: "Churn"})
    )


# ==================================================
# 3. MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.title("🤖 XGBoost Model Performance")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", "95.26%")
    col2.metric("Precision", "87.80%")
    col3.metric("Recall", "80.90%")
    col4.metric("F1 Score", "84.21%")
    col5.metric("ROC-AUC", "98.00%")

    st.subheader("Model Information")

    st.write("""
    The customer churn model uses XGBoost with preprocessing
    for numerical and categorical features.

    The model was evaluated on a held-out test set.

    The classification threshold of 0.50 was retained because
    it produced the highest F1 score among the tested thresholds.
    """)


# ==================================================
# 4. NEW CUSTOMER PREDICTION
# ==================================================

elif page == "New Customer Prediction":

    st.title("🔮 New Customer Churn Prediction")

    st.write(
        "Enter customer information to estimate the probability of churn."
    )

    col1, col2 = st.columns(2)

    with col1:

        call_failure = st.number_input(
            "Call Failure",
            min_value=0,
            value=5
        )

        complains = st.selectbox(
            "Complains",
            [0, 1]
        )

        subscription_length = st.number_input(
            "Subscription Length",
            min_value=0,
            value=12
        )

        charge_amount = st.number_input(
            "Charge Amount",
            min_value=0,
            value=5
        )

        seconds_of_use = st.number_input(
            "Seconds of Use",
            min_value=0,
            value=2000
        )

        frequency_use = st.number_input(
            "Frequency of Use",
            min_value=0,
            value=20
        )

        frequency_sms = st.number_input(
            "Frequency of SMS",
            min_value=0,
            value=10
        )

    with col2:

        distinct_numbers = st.number_input(
            "Distinct Called Numbers",
            min_value=0,
            value=10
        )

        age_group = st.number_input(
            "Age Group",
            min_value=0,
            value=3
        )

        tariff_plan = st.selectbox(
            "Tariff Plan",
            [1, 2]
        )

        status = st.selectbox(
            "Status",
            [1, 2]
        )

        age = st.number_input(
            "Age",
            min_value=0,
            value=30
        )

        customer_value = st.number_input(
            "Customer Value",
            min_value=0.0,
            value=500.0
        )

    # ----------------------------------------------
    # Create input dataframe
    # ----------------------------------------------

    new_customer = pd.DataFrame([{
        "Call  Failure": call_failure,
        "Complains": complains,
        "Subscription  Length": subscription_length,
        "Charge  Amount": charge_amount,
        "Seconds of Use": seconds_of_use,
        "Frequency of use": frequency_use,
        "Frequency of SMS": frequency_sms,
        "Distinct Called Numbers": distinct_numbers,
        "Age Group": age_group,
        "Tariff Plan": tariff_plan,
        "Status": status,
        "Age": age,
        "Customer Value": customer_value
    }])

    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    if st.button("Predict Churn"):

        prediction = model.predict(new_customer)[0]
        probability = model.predict_proba(new_customer)[0, 1]

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

        if prediction == 1:
            col2.error("⚠️ Customer Predicted to Churn")
        else:
            col2.success("✅ Customer Predicted Not to Churn")

        # Risk interpretation
        if probability < 0.30:
            risk = "Low"
        elif probability < 0.60:
            risk = "Medium"
        else:
            risk = "High"

        st.info(f"Risk Level: **{risk}**")