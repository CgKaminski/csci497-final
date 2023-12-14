import pandas as pd
from sklearn.cross_decomposition import PLSRegression
import numpy as np

inputs = pd.read_csv('data/preprocessed_input.csv')
targets = pd.read_csv('data/targets.csv')

X = inputs.values[:, 1:]
Y = targets.values[:, 2:]


pls = PLSRegression(n_components=21)
pls.fit(X, Y)

Y_pred = pls.predict(X)

correlation_coeffs = [np.corrcoef(Y[:, i], Y_pred[:, i])[0, 1] for i in range(Y.shape[1])]
print('Correlation coefficients for each response variable:', correlation_coeffs)