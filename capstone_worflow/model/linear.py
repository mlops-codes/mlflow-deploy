from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def build():
    return Pipeline([("scaler", StandardScaler()),
                     ("regressor", LinearRegression())])
