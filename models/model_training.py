from sklearn.linear_model import LogisticRegression
import pickle

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

def export_file(model, scaler):
    """
    Exporting model and scaler to binary files.

    Parameters:
    - model: Trained model to be serialized.
    - scaler: Trained scaler to be serialized.
    """
    model_path = 'artifacts/model.pkl'
    scaler_path = 'artifacts/scaler.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
