import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the trained model and encoders
model = joblib.load('cancer_model.pkl')
encoder = joblib.load('encoder.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = StandardScaler()


# Streamlit app
st.title('Cancer Diagnosis Prediction')

# Input form
age = st.number_input('Age', min_value=0)
sex = st.selectbox('Sex', ['M', 'F'])
county = st.selectbox('County', ['Homabay', 'Migori'])  # Add other counties as needed
hiv_status = st.selectbox('HIV Status', [1, 2], format_func=lambda x: 'Positive' if x == 1 else 'Negative')

# Predict button
if st.button('Predict'):
    # Create a DataFrame for the new data
    data = {'AGE': [age], 'SEX': [sex], 'COUNTY': [county], 'HIV STATUS': [hiv_status]}
    new_df = pd.DataFrame(data)

    # Preprocess the new data
    encoded_features = encoder.transform(new_df[['SEX', 'COUNTY']]).toarray()
    encoded_df = pd.DataFrame(encoded_features)
    new_df = pd.concat([new_df, encoded_df], axis=1)
    new_df.drop(['SEX', 'COUNTY'], axis=1, inplace=True)
    new_df[['AGE']] = scaler.fit_transform(new_df[['AGE']])

    # Convert column names to strings
    new_df.columns = new_df.columns.astype(str)

    # Make prediction
    prediction = model.predict(new_df)

    # Decode the prediction
    predicted_diagnosis = label_encoder.inverse_transform([round(prediction[0])])[0]

    # Display prediction
    st.success(f'Predicted Diagnosis: {predicted_diagnosis}')

# Cancer Distribution Visualization with Line Graph
st.header('Cancer Distribution')



fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(diagnosis_counts.index, diagnosis_counts.values, marker='o', linestyle='-')
ax.set_xlabel('Diagnosis')
ax.set_ylabel('Number of Cases')
ax.set_title('Cancers Affecting More People')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)
