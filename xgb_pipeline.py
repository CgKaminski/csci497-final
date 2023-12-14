import numpy as np
import pandas as pd
import xgboost as xg
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def load_data() -> tuple:
  """
  Load the preprocessed data from the data directory.
  :return: tuple of (X_train, y_train, X_dev, y_dev)
  """
  inputs = pd.read_csv('data/preprocessed_input.csv')
  targets = pd.read_csv('data/targets.csv')
  X = inputs.values
  Y = targets.values[:, 1:]
  val_idx = np.argmax(targets.values[:, 0] >= 600)

  X_train = X[:val_idx, :]
  y_train = Y[:val_idx, :]
  X_dev = X[val_idx:, :]
  y_dev = Y[val_idx:, :]

  return X_train, y_train, X_dev, y_dev


def xgb_pipeline(seed=19, learning_rate=0.1, max_depth=5, reg_alpha=1, reg_lambda=0.01, n_est=10) -> Pipeline:
  """
  Create an XGBoost pipeline.
  :param seed: Random seed for reproducibility.
  :param learning_rate: Step size shrinkage to prevent overfitting.
  :param max_depth: Maximum depth of a tree.
  :param reg_alpha: L1 regularization term.
  :param reg_lambda: L2 regularization term.
  :param n_est: Number of boosting rounds (trees).
  :return: XGBoost pipeline.
  """
  xgb_r = xg.XGBRegressor(objective='reg:squarederror', max_depth=max_depth,
                          learning_rate=learning_rate, reg_alpha=reg_alpha,
                          reg_lambda=reg_lambda, seed=seed, n_estimators=n_est)
  return Pipeline([('xgb', xgb_r)])


def xgb_hyperparam_search(X, y, param_dist, n_iter=10, cv=5, scoring='neg_mean_squared_error', seed=42):
  """
  Perform hyperparameter search for XGBoost using RandomizedSearchCV.
  :param X: Input features.
  :param y: Target values.
  :param param_dist: Hyperparameter search space.
  :param n_iter: Number of random combinations to try.
  :param cv: Number of cross-validation folds.
  :param scoring: Scoring metric for optimization.
  :param seed: Random seed for reproducibility.
  :return: Best hyperparameters and corresponding score.
  """
  xgb_pipe = xgb_pipeline()

  # Create the RandomizedSearchCV object
  random_search = RandomizedSearchCV(
    xgb_pipe,
    param_distributions=param_dist,
    n_iter=n_iter,
    cv=cv,
    verbose=3,
    scoring=scoring,
    random_state=seed
  )

  # Fit the RandomizedSearchCV object to the data
  random_search.fit(X, y)

  # Print the best set of hyperparameters and the corresponding score
  print("Best set of hyperparameters: ", random_search.best_params_)
  print("Best score: ", random_search.best_score_)

  return random_search.best_params_, random_search.best_score_


def main():
  X_train, y_train, X_dev, y_dev = load_data()

  # Define the hyperparameter grid for a sweep
  param_dist = {
    'xgb__max_depth': [3, 5, 10],
    'xgb__learning_rate': [0.01, 0.1, 1],
    'xgb__reg_lambda': [0, 0.01, 0.1, 1, 10],
    'xgb__reg_alpha': [0, 0.01, 0.1, 1, 10],
    'xgb__n_estimators': [10, 50, 100, 200]
  }

  best_params, best_score = xgb_hyperparam_search(X_train, y_train[:, 0], param_dist)

  # Use the best hyperparameters to fit the final model
  final_model = xgb_pipeline(**best_params)
  final_model.fit(X_train, y_train[:, 0])

  # Predictions on development set
  Y_dev_pred = final_model.predict(X_dev)
  print('\nDevelopment Set Metrics:')
  print('R-squared:', r2_score(y_dev[:, 0], Y_dev_pred))


if __name__ == '__main__':
  main()
