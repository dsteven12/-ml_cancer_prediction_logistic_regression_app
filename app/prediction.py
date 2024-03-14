import pickle
import pandas as pd
import streamlit as st

def load_model_scaler_features():
    """
    Load the trained model, scaler, and feature names from disk.

    Returns:
    - model: The trained Logistic Regression model.
    - scaler: The StandardScaler used for data normalization.
    - features: The feature names used during training.
    """
    with open('artifacts/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('artifacts/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('artifacts/features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    return model, scaler, features

def add_predictions(input_data):
    """
    Adds prediction output to the Streamlit app.

    Parameters:
    - input_data (dict): The user input data for prediction.
    """
    model, scaler, features = load_model_scaler_features()

    # Construct a DataFrame from input_data using the ordered features list
    # This ensures the DataFrame matches the training structure
    input_df = pd.DataFrame([input_data], columns=features)
    input_array_scaled = scaler.transform(input_df)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)

    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malignant: ", model.predict_proba(input_array_scaled)[0][1])
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")