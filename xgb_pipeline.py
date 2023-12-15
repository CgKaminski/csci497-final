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


def xgb_pipeline(X_train: np.ndarray, y_train: np.ndarray, X_dev: np.ndarray) -> xg.XGBRegressor:
  model = xg.XGBRegressor(objective='reg:squarederror', n_estimators=10, random_state=19, learning_rate=0.1, 
                          max_depth=5, reg_alpha=1, reg_lambda=0.01)
  param_grid = {
    'n_estimators': [1, 5, 10, 25, 50, 100],
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [1, 2, 3, 4, 5, 10, 20, 30],
    'reg_alpha': [1, 0.1, 0.01],
    'reg_lambda': [1, 0.1, 0.01]
  }

  # RandomizedSearchCV
  random_search = RandomizedSearchCV(model, param_grid, n_iter=25, cv=3, verbose=3, random_state=19, n_jobs=-1)
  random_search.fit(X_train, y_train)

  return random_search.best_estimator_


    
def main() -> None:
  X_train, y_train, X_dev, y_dev = load_data()
  model = xgb_pipeline(X_train, y_train, X_dev)
  y_pred = model.predict(X_dev)
  print(f'R2 score: {r2_score(y_dev, y_pred)}')

if __name__ == '__main__':
  main()
