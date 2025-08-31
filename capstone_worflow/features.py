import pandas as pd

def split_xy(df: pd.DataFrame):
    X = df[["YearsExperience"]]
    y = df["Salary"]
    return X, y
