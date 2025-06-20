import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the page configuration
st.set_page_config(page_title="Car Price Prediction App", page_icon="🚗", layout="wide")

# Loading the trained model
model = joblib.load('car_price_model.pkl')

# Loading the dataset to get unique values for dropdowns
data = pd.read_csv('cardataformodel.csv')

# Getting unique values for dropdowns
makes = data['Make'].unique().tolist()
colours = data['Colour'].unique().tolist()

# App title and description
st.title("🚗 Used Car Price Prediction")
st.markdown("""
This app predicts the price of a used car based on its features. Enter the details below and click **Predict** to see the estimated price.
The model uses a Random Forest Regressor trained on a dataset of used cars.
""")

# Creating input form
st.header("Enter Car Details")
with st.form(key="car_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        make = st.selectbox("Make", makes, help="Select the car manufacturer")
        colour = st.selectbox("Colour", colours, help="Select the car colour")
        odometer = st.number_input("Odometer (KM)", min_value=0, max_value=500000, value=50000, step=1000, help="Enter the car's mileage in kilometers")
    
    with col2:
        doors = st.slider("Doors", min_value=2, max_value=5, value=4, step=1, help="Number of doors")
        seats = st.slider("Seats", min_value=2, max_value=8, value=5, step=1, help="Number of seats")
        fuel_milage = st.number_input("Fuel Mileage (L/100km)", min_value=0.0, max_value=20.0, value=5.0, step=0.1, help="Fuel consumption in liters per 100km")
        total_fuel_used = st.number_input("Total Fuel Used (L)", min_value=0, max_value=100000, value=10000, step=100, help="Total fuel used in liters")
    
    submit_button = st.form_submit_button("Predict Price")

# Prediction logic
if submit_button:
    # Creating input DataFrame
    input_data = pd.DataFrame({
        'Make': [make],
        'Colour': [colour],
        'Odometer (KM)': [odometer],
        'Doors': [doors],
        'Seats': [seats],
        'Fuel Milage': [fuel_milage],
        'Total Fuel Used': [total_fuel_used]
    })
    
    # Making prediction
    try:
        predicted_price = model.predict(input_data)[0]
        st.success(f"**Predicted Price**: ${predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Feature importance visualization
st.header("Model Insights")
st.markdown("The following plot shows the importance of each feature in predicting the car price.")
feature_names = (['Odometer (KM)', 'Doors', 'Seats', 'Fuel Milage', 'Total Fuel Used'] + 
                 model.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .get_feature_names_out(['Make', 'Colour']).tolist())
importances = model.named_steps['regressor'].feature_importances_

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names, ax=ax)
ax.set_title('Feature Importance in Car Price Prediction')
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
st.pyplot(fig)

