import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score


def load_data() -> tuple:
  """
  Load the preprocessed data from the data directory.
  :return: tuple of (X_train, y_train, X_dev, y_dev)
  """
  inputs = pd.read_csv('data/preprocessed_input.csv')
  targets = pd.read_csv('data/targets.csv')
  X = inputs.values
  y = targets.values[:, 1:]
  val_idx = np.argmax(targets.values[:, 0] >= 600)

  X_train = X[:val_idx, :]
  y_train = y[:val_idx, :]
  X_dev = X[val_idx:, :]
  y_dev = y[val_idx:, :]

  return X, y, X_train, y_train, X_dev, y_dev


from sklearn.metrics import mean_squared_error

def pls_components(y, X_train, y_train, X_dev, y_dev, n_components: int) -> tuple:
    """
    Fit a PLS model with n_components and print the PRESS and r2 scores for the training and validation sets.
    :param y:
    :param X_train:
    :param y_train:
    :param X_dev:
    :param y_dev:
    :param n_components:
    :return:
    """

    train_press = np.zeros((n_components, y.shape[1]))
    train_r2 = np.zeros((n_components, y.shape[1]))
    val_press = np.zeros((n_components, y.shape[1]))
    val_r2 = np.zeros((n_components, y.shape[1]))

    for i in range(1, n_components):
        print(f'Fitting PLS with {i} components')
        pls = PLSRegression(n_components=i)
        pls.fit(X_train, y_train)
        y_pred_train = pls.predict(X_train)
        train_press[i, :] = [np.sum((y_train[:, j] - y_pred_train[:, j])**2) for j in range(y.shape[1])]
        train_r2[i, :] = [r2_score(y_train[:, j], y_pred_train[:, j]) for j in range(y.shape[1])]

        y_pred_dev = pls.predict(X_dev)
        val_press[i, :] = [np.sum((y_dev[:, j] - y_pred_dev[:, j])**2) for j in range(y.shape[1])]
        val_r2[i, :] = [r2_score(y_dev[:, j], y_pred_dev[:, j]) for j in range(y.shape[1])]

    return train_press, train_r2, val_press, val_r2



def main():
  X, y, X_train, y_train, X_dev, y_dev = load_data()
  train_corr, train_r2, val_corr, val_r2 = pls_components(y, X_train, y_train, X_dev, y_dev, 30)

  # Save all the results together on a single csv
  results = np.concatenate((train_corr, train_r2, val_corr, val_r2), axis=1)
  np.savetxt('output/pls_components.csv', results, delimiter=',')


if __name__ == '__main__':
  main()
