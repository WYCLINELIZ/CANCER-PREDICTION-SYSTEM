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

# Load the original dataset
df = pd.read_csv('CANCER.csv')

# User Authentication (Simulated user database)
if 'users' not in st.session_state:
    st.session_state.users = {"user1": "password1", "user2": "password2"}  # Replace with a secure method

# Session state for login status
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Navbar
def navbar():
    selected = st.sidebar.selectbox("Navigate", ["Home", "Services", "Outcomes", "Logout"])
    return selected

# Combined Signup/Login Page
def auth_page():
    st.image("cancer (3).jpeg", width=250)
    
    if st.session_state.signup_success:
        st.success("Signup successful! You can now log in with your credentials.")
    
    st.title('Cancer Diagnosis Prediction System')
    
    # Login Form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid username or password")
            
    # Signup Form
    new_username = st.text_input("New Username", key="new_username")
    new_password = st.text_input("New Password", type="password", key="new_password")

    if st.button("Sign Up"):
        if new_username in st.session_state.users:
            st.error("Username already exists. Please choose a different username.")
        else:
            # Add new user to the simulated user database
            st.session_state.users[new_username] = new_password
            st.session_state.signup_success = True  # Flag for signup success
            st.session_state.new_username = ""  # Clear the new username field

# Cancer Diagnosis Prediction
def prediction_page():
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

# Cancer Distribution Visualization
def outcomes_page():
    st.header('Cancer Distribution')

    diagnosis_counts = df['DIAGNOSIS'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(diagnosis_counts.index, diagnosis_counts.values, marker='o', linestyle='-')
    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Number of Cases')
    ax.set_title('Cancers Affecting More People')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# Main app logic
if st.session_state.logged_in:
    page = navbar()

    if page == "Home":
        prediction_page()
    elif page == "Services":
        st.write("Services page (can add more information here)")
    elif page == "Outcomes":
        outcomes_page()
    elif page == "Logout":
        st.session_state.logged_in = False
        st.session_state.signup_success = False  # Reset signup success status
        st.success("You have successfully logged out.")
else:
    # Initialize signup success flag
    if 'signup_success' not in st.session_state:
        st.session_state.signup_success = False
    auth_page()
