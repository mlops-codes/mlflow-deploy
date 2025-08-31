from sklearn.tree import DecisionTreeRegressor

def build(random_state=42):
    return DecisionTreeRegressor(random_state=random_state)
