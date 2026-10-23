# ChurnIQ — Customer Churn Prediction

## Overview
ChurnIQ is a machine learning web application that predicts the likelihood of a customer leaving a service. It helps businesses identify customers who may be at risk and take proactive steps to improve customer retention.

## Features
- Predicts customer churn using a trained Logistic Regression model.
- Accepts customer information through an interactive web interface.
- Displays estimated churn probability.
- Categorizes customers into low, moderate, and high risk.
- Provides a simple dashboard built with Streamlit.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Machine Learning

## Machine Learning Model
The application uses a trained Logistic Regression pipeline to process customer details and generate a churn prediction. The displayed probability and risk category are estimates, not guarantees of future customer behavior.

## Project Structure
```text
Customer-Churn-Prediction/
├── app.py
├── churn_model_new.pkl
├── requirements.txt
└── README.md
```

## Run Locally

1. Clone this repository:
   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```
2. Open the project folder:
   ```bash
   cd Customer-Churn-Prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the application:
   ```bash
   python -m streamlit run app.py
   ```

## Future Improvements
- Add interactive visualizations.
- Evaluate additional machine learning models.
- Provide more detailed customer insights.
- Deploy the application online.

## Disclaimer
Predictions are estimates based on the trained model and should support—not replace—business judgment.
