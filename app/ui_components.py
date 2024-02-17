import streamlit as st

def load_style(style_path):
    """
    Loads and applies CSS styles from a given file.
    
    Parameters:
    - style_path (str): Path to the CSS file.
    """
    try:
        with open(style_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Style configuration file not found.")

def render_title_and_description():
    """
    Renders the title and description of the app in the Streamlit UI.
    """
    st.title("Breast Cancer Predictor")
    st.write(
        "Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample."
        " This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab."
        " You can also update the measurements by hand using the sliders in the sidebar."
    )