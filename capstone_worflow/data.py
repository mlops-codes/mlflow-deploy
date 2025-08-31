import pandas as pd
from config import settings

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(settings.data_path)
    if df.columns[0].lower().startswith("unnamed") or df.columns[0] == "":
        df = df.iloc[:, 1:]
    return df[["YearsExperience", "Salary"]]
