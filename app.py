import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("/content/rfr_model")

# Define the function to preprocess user input
def preprocess_input(age, sex, bmi, children, smoker):
    # Perform data preprocessing
    data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker]
    })
    return data

# Define the function to predict insurance charge
def predict_insurance_charge(data):
    # Use the loaded model to make predictions
    prediction = model.predict(data)
    return prediction

# Create the Streamlit web app
def main():
    # Set the title and sidebar
    st.title("Insurance Charge Estimation")
    st.sidebar.title("User Input")

    # Add sliders/inputs for user input
    age = st.sidebar.slider("Age", 20, 100, step=1, value=30)
    sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    bmi = st.sidebar.slider("BMI", 10.0, 40.0, step=0.1, value=20.0)
    children = st.sidebar.slider("Number of Children", 0, 10, step=1, value=0)
    smoker = st.sidebar.selectbox("Smoker", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    # Preprocess the input
    input_data = preprocess_input(age, sex, bmi, children, smoker)

    # Make predictions
    prediction = predict_insurance_charge(input_data)

    # Display the prediction
    st.subheader("Estimated Insurance Charge:")
    result_placeholder = st.empty()
    result_placeholder.write(prediction[0])


if __name__ == "__main__":
    # Launch the app in Streamlit
    main()
