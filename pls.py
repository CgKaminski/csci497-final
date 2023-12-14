import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
import numpy as np

inputs = pd.read_csv('data/preprocessed_input.csv')
targets = pd.read_csv('data/targets.csv')

X = inputs.values[:, 1:]
Y = targets.values[:, 2:]

val_idx = np.argmax(targets.values[:, 1] >= 600)

X_train = X[:val_idx, :]
Y_train = Y[:val_idx, :]
X_val = X[val_idx:, :]
Y_val = Y[val_idx:, :]

mse = []
correlations = []
n_components = np.arange(21, 22)
for i in n_components:
    pls = PLSRegression(n_components=i)
    pls.fit(X_train, Y_train)

    Y_pred = pls.predict(X_train)
    t_corr = [np.corrcoef(Y_train[:, j], Y_pred[:, j])[0, 1] for j in range(Y.shape[1])]
    t_r2 = [r2_score(Y_train[:, j], Y_pred[:, j]) for j in range(Y.shape[1])]
    print(t_corr)
    print(t_r2)

    Y_pred = pls.predict(X_val)
    v_corr = [np.corrcoef(Y_val[:, j], Y_pred[:, j])[0, 1] for j in range(Y.shape[1])]
    v_r2 = [r2_score(Y_val[:, j], Y_pred[:, j]) for j in range(Y.shape[1])]
    print(v_corr)
    print(v_r2)
