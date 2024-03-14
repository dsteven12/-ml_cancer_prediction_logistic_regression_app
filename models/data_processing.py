import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath):
    """
    Load data from a CSV file and clean it by removing specific columns and mapping diagnosis values.
    
    Parameters:
    - filepath (str): The path to the CSV file.
    
    Returns:
    - pd.DataFrame: The cleaned data with features and labels.
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None, None, None

    data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    return X, y, data

def normalize_data(X):
    """
    Normalize the feature data using StandardScaler.
    
    Parameters:
    - X (pd.DataFrame): The feature data to be normalized.
    
    Returns:
    - np.ndarray: The normalized feature data.
    - StandardScaler: The scaler used for normalization.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def split_data(X, y, test_size=0.3, random_state=42):
    """
    Split the data into training and testing sets.
    
    Parameters:
    - X (np.ndarray): Normalized feature data.
    - y (pd.Series): Labels.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Controls the shuffling applied to the data before applying the split.
    
    Returns:
    - Tuple containing split data (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)