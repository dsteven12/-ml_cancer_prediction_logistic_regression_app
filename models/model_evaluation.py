from sklearn.metrics import accuracy_score, classification_report
import pickle

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