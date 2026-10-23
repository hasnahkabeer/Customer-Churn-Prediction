
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="ChurnIQ | Customer Analytics",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("churn_model_new.pkl")

model = load_model()

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}
.hero {
    background: linear-gradient(120deg, #14213d, #2456a6);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 8px;
}
.hero p {
    color: #e0eaff;
    font-size: 16px;
}
div[data-testid="stForm"] {
    background: white;
    padding: 24px;
    border: 1px solid #e3e9f2;
    border-radius: 16px;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e3e9f2;
}
h2, h3 {
    color: #14213d;
}
.stButton button,
div[data-testid="stFormSubmitButton"] button {
    background: #2456a6;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
}
.footer {
    text-align: center;
    color: #738096;
    font-size: 13px;
    padding-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DASHBOARD HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>📊 ChurnIQ</h1>
    <p>Customer Retention Intelligence Dashboard</p>
    <p>Understand customer churn risk with machine learning.</p>
</div>
""", unsafe_allow_html=True)

st.subheader("👤 Customer Profile")
st.write(
    "Enter the customer's information below to estimate their "
    "likelihood of leaving the service."
)

# ---------------- CUSTOMER INPUT FORM ----------------
with st.form("customer_form"):

    st.markdown("#### 👤 Personal Information")
    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])

    with c2:
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with c3:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    st.markdown("#### 💳 Subscription & Billing")
    c4, c5, c6 = st.columns(3)

    with c4:
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with c5:
        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=5.0
        )
        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with c6:
        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
            step=10.0
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.markdown("#### 🌐 Internet & Services")
    c7, c8, c9 = st.columns(3)

    with c7:
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

    with c8:
        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

    with c9:
        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    submitted = st.form_submit_button(
        "🔍 Analyze Customer Churn",
        use_container_width=True
    )

# ---------------- PREDICTION & RESULTS ----------------
if submitted:

    customer = pd.DataFrame([{
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment
    }])

    prediction = model.predict(customer)[0]
    probabilities = model.predict_proba(customer)[0]

    # Get probability for class 1 (confirm class 1 means churn
    # in the target labels used to train your model).
    churn_probability = float(
        probabilities[list(model.classes_).index(1)]
    )

    st.markdown("---")
    st.subheader("📈 Churn Analysis Results")

    r1, r2, r3 = st.columns(3)

    with r1:
        outcome = (
            "Churn Risk" if prediction == 1
            else "Likely to Stay"
        )
        st.metric("Predicted Outcome", outcome)

    with r2:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )

    with r3:
        risk_category = (
            "High" if churn_probability >= 0.60
            else "Moderate" if churn_probability >= 0.30
            else "Low"
        )
        st.metric("Risk Category", risk_category)

    st.progress(churn_probability)

    # ---------------- CUSTOMER RISK SUMMARY ----------------
    st.markdown("### 📝 Customer Risk Summary")

    if churn_probability >= 0.60:
        st.warning(
            "This customer has a higher estimated churn probability. "
            "Consider reviewing their service experience and "
            "retention options."
        )
    elif churn_probability >= 0.30:
        st.info(
            "This customer has a moderate estimated churn probability. "
            "Consider monitoring satisfaction and reviewing their "
            "subscription."
        )
    else:
        st.success(
            "This customer has a lower estimated churn probability. "
            "Continue providing good service and monitoring "
            "customer satisfaction."
        )

    st.caption(
        "Risk categories use illustrative probability thresholds. "
        "The model estimates churn likelihood; it does not guarantee "
        "future customer behavior or establish why a customer may leave."
    )

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    ChurnIQ · Machine Learning Customer Analytics · Built with Streamlit
</div>
""", unsafe_allow_html=True)

