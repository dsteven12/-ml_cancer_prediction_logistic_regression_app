from models import load_and_clean_data, normalize_data, split_data, train_model, evaluate_model, export_file

def main():
    filepath = "data/data.csv"  # Adjust the path as necessary
    X, y = load_and_clean_data(filepath)
    if X is not None:
        X_scaled, scaler = normalize_data(X)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        export_file(model, scaler)

if __name__ == "__main__":
    main()
