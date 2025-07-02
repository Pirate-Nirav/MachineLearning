import streamlit as st
import pandas as pd
import joblib
pipeline = joblib.load("fare_prediction_model.pkl")


st.set_page_config(page_title="Taxi Fare Prediction", page_icon="🚕", layout="centered")

st.title("🚕 Taxi Fare Prediction App")
st.write("""
Enter the details of your trip to predict the expected fare.
""")

st.header("📋 Input Trip Details")

col1, col2 = st.columns(2)

with col1:
    trip_miles = st.number_input("Trip Miles", min_value=0.0, step=0.1)
    trip_seconds = st.number_input("Trip Seconds", min_value=0, step=10)
    tip_rate = st.number_input("Tip Rate", min_value=0.0, max_value=1.0, step=0.01)

with col2:
    # Replace these with the actual unique values from your data
    companies = ['Company A', 'Company B', 'Company C']
    payment_types = ['Cash', 'Card', 'Mobile']

    company = st.selectbox("Company", companies)
    payment_type = st.selectbox("Payment Type", payment_types)


if st.button("Predict Fare"):
    # Prepare input DataFrame
    input_df = pd.DataFrame({
        'TRIP_MILES': [trip_miles],
        'TRIP_SECONDS': [trip_seconds],
        'COMPANY': [company],
        'PAYMENT_TYPE': [payment_type],
        'TIP_RATE': [tip_rate]
    })

    # Make prediction
    predicted_fare = pipeline.predict(input_df)[0]

    st.success(f"💰 Estimated Fare: **${predicted_fare:.2f}**")

st.markdown("""

""")
