import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def get_clean_data():
    data = pd.read_csv("../data/data.csv")
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0})

    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    return X, y

def normalize_data(X, y):
    # Normalize Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def split_data(X, y):
    # Split the Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # Train
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def test_model(model, X_test, y_train, y_test):   
    # Test Model
    y_pred = model.predict(X_test)
    print(f'Accuracy of our model: {accuracy_score(y_test, y_pred)}')
    print(f"Classification report: \n {classification_report(y_test, y_pred)}")

    return None


def main():
    X, y = get_clean_data()
    X_scaled, scaler = normalize_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    model = train_model(X_train, y_train)
    test_model(model, X_test, y_train, y_test)


if __name__ == "__main__":
    main()