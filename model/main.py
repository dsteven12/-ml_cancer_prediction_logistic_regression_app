import pandas as pd

def get_clean_data():
    data = pd.read_csv("../data/data.csv")
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0})
    
    return data

def main():
    data = get_clean_data()

if __name__ == "__main__":
    main()