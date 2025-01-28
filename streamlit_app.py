import streamlit as st
import pickle
import numpy as np

# Load the SVM model and standard scaler
svm_model = pickle.load(open('svm.pkl', 'rb'))
standard_scaler = pickle.load(open('scaler.pkl', 'rb'))

# Streamlit app
st.title("Chronic Kidney Disease Prediction")
st.write("Enter the following details to predict the presence of Chronic Kidney Disease:")

# Input fields
age = st.number_input("Age", min_value=0.0, step=1.0)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, step=1.0)
specific_gravity = st.number_input("Specific Gravity", min_value=0.0, step=0.01, format="%.2f")
albumin = st.number_input("Albumin", min_value=0.0, step=0.1)
sugar = st.number_input("Sugar", min_value=0.0, step=0.1)
red_blood_cells = st.selectbox("Red Blood Cells Condition", options=["normal", "abnormal"])
hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, step=0.1)
packed_cell_volume = st.number_input("Packed Cell Volume", min_value=0.0, step=1.0)

# Predict button
if st.button("Predict"):
    try:
        # Collect input data and preprocess
        data = [
            age,
            blood_pressure,
            specific_gravity,
            albumin,
            sugar,
            1.0 if red_blood_cells == "normal" else 0.0,
            hemoglobin,
            packed_cell_volume
        ]
        input_data = np.array(data).reshape(1, -1)
        scaled_data = standard_scaler.transform(input_data)

        # Make prediction
        prediction = svm_model.predict(scaled_data)
        result = "No Chronic Kidney Disease" if prediction[0] == 0 else "Chronic Kidney Disease Detected"

        # Display result
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
