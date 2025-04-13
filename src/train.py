from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from src.models import XGBoostModel, SquaredErrorObjective

if __name__ == '__main__':
    X, y = fetch_california_housing(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

    hyperparams = {
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'reg_lambda': 1.5,
        'gamma': 0.0,
        'min_child_weight': 25,
        'base_score': 0.0,
        'tree_method': 'exact',
    }

    model = XGBoostModel(params=hyperparams, seed=42)
    model.fit(X_train, y_train, objective=SquaredErrorObjective(), num_boost_rounds=50, verbose=True)

    predictions = model.predict(X_test)
    print(f'scratch score: {SquaredErrorObjective().loss(y_test, predictions)}')
