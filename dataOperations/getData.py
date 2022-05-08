import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path).head(50000).dropna()

def split_dataset(X_data: pd.DataFrame, y_data: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(X_data.values, y_data.values, test_size=0.3, random_state=0)
    return {'X_train': X_train, 'X_test': X_test, 'Y_train': y_train, 'Y_test': y_test}