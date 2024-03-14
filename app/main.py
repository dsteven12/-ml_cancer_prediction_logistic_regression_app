import sys
import os

# Ensure that the parent directory is recognized for module imports,
# allowing imports from the 'models' directory.
current_script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)

from models.data_processing import load_and_clean_data
import streamlit as st
from app import add_sidebar, get_radar_chart, add_predictions, load_style, render_title_and_description

def main():
    """
    Main function to launch the Streamlit app for breast cancer prediction.
    Loads data, configures the app's page, and renders UI components.
    """
    filepath = "../data/data.csv"
    # Load and preprocess data for the app.
    X, _, data = load_and_clean_data(filepath)

    # Configure Streamlit app appearance and initial settings.
    st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
    )

    # Apply custom styles from external CSS.
    load_style("../assets/style.css")

    # Collect user input for prediction via sidebar sliders.
    input_data = add_sidebar(data)

    # Main content area: visualization and prediction results.
    render_title_and_description()
    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data, X)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)

if __name__ == '__main__':
    main()