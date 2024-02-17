import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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
        return None, None

    data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    return X, y

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

def train_model(X_train, y_train):
    """
    Train a Logistic Regression model on the training data.
    
    Parameters:
    - X_train (np.ndarray): Training feature data.
    - y_train (pd.Series): Training labels.
    
    Returns:
    - LogisticRegression: The trained model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data and print accuracy and classification report.
    
    Parameters:
    - model (LogisticRegression): The trained model.
    - X_test (np.ndarray): Testing feature data.
    - y_test (pd.Series): Testing labels.
    """
    y_pred = model.predict(X_test)
    print(f'Accuracy of our model: {accuracy_score(y_test, y_pred)}')
    print(f"Classification report: \n {classification_report(y_test, y_pred)}")

def main():
    """
    Main function to run the data preparation, model training, and evaluation pipeline.
    """
    filepath = "../data/data.csv"  # Consider making this an argument or a configurable variable
    X, y = load_and_clean_data(filepath)
    if X is not None:
        X_scaled, scaler = normalize_data(X)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
