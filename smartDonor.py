import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

# Loading the saved XGBoost model and the OneHotEncoder
model = joblib.load('xgboost_model_donor.pkl')
encoder = joblib.load('onehot_encoder_donor.pkl') 

# Title
st.title("Smart Donor App")
st.write("This app predicts the likelihood of a potential blood donor donating for a patient in need of transfusion.")


# User input features
gender = st.selectbox("Gender", options=["Male", "Female"])
blood_group = st.selectbox("Blood Group", options=["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
last_donation_nature = st.selectbox("Nature of Last Donation", options=["Voluntary", "Replacement"])
reaction = st.selectbox("Any reaction from previous donation?", options=["No", "Yes"])
convenient_time = st.selectbox("Convenient time to donate", options=["Morning", "Afternoon", "Evening", "Anytime"])
convenient_locality = st.selectbox("Convenient locality to donate", options=["Anywhere", "Mvita", "Kisauni", "Nyali", "Changamwe", "Likoni", "Jomvu", "Mombasa"])
encouragement = st.selectbox("Encouragement to donate", options=["Self", "Friends", "Family"])

# Numerical features
times_donated = st.number_input("Times Donated", min_value=0, max_value=100, value=5)
preferred_freq = st.number_input("Preferred Donation Frequency (1-4)", min_value=1, max_value=4, value=3)
rate_service = st.slider("Rate Last Donation Service (1-10)", min_value=1, max_value=10, value=8)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
days_since_last = st.number_input("Days Since Last Donation", min_value=0, max_value=365, value=120)

# When the "Predict" button is clicked
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Blood_Group': [blood_group],
        'Nature_Last_Donation': [last_donation_nature],
        'Times_Donated': [times_donated],
        'Preffered_Freq': [preferred_freq],
        'Rate_Service': [rate_service],
        'Reaction': [reaction],
        'Convenient_Time': [convenient_time],
        'Convenient_Locality': [convenient_locality],
        'Encouragement': [encouragement],        
        'Age': [age],
        'DaysSinceLastDonation': [days_since_last]
    })

    # Apply One-Hot Encoding to the categorical data
    input_data_encoded = encoder.transform(input_data)

    # Predict the likelihood of donation
    prediction = model.predict(input_data_encoded)
    
    # Display the prediction result
    #st.write(f"Predicted likelihood of donation: {prediction[0]:.2f}")
    
    st.markdown(
        f"<h1 style='color: #FF1493; font-weight: bold; font-size: 36px;'>Predicted likelihood of donation: {prediction[0]:.2f}</h1>",
        unsafe_allow_html=True
    )
