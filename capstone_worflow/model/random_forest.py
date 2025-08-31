from sklearn.ensemble import RandomForestRegressor

def build(n_estimators=200, random_state=42):
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
