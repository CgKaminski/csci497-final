from dataset import load_data
from preprocessing import *
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


# generate data if not loading from a file
'''
fs = 1000 # sampling freq (Hz)
ecog_data, motion_data = load_data()

for key in tqdm(ecog_data.keys()[1:], desc='Band Pass Filter'):
    ecog_data[key] = band_pass_filter(ecog_data[key].values)

filt_ecog_data = ecog_data.values[:,1:]
car_ecog_data = car(filt_ecog_data)

hand_data, time = downsample(motion_data[['MotionTime','Experimenter:RHNDx','Experimenter:RHNDy','Experimenter:RHNDz']])


hand_df = pd.DataFrame(hand_data, columns=['Time', 'Hand:x', 'Hand:y', 'Hand:z'])
hand_df = hand_df[hand_df['Time'] > 1.1]

# normalize hand data
normalized = normalize_targets(hand_df.to_numpy()[:, 1:])
hand_df['Hand:x'] = normalized[:, 0]
hand_df['Hand:y'] = normalized[:, 1]
hand_df['Hand:z'] = normalized[:, 2]
hand_df.to_csv('data/targets.csv', index=False)




input_dataset = []
for start_time in tqdm(hand_df['Time'].values, desc='Wavelet Transform for all time steps'):
    neuron_data = []
    for neuron in range(64):
        start_index = ecog_data.index[ecog_data['ECoG_time'] == start_time][0]
        batch = car_ecog_data[:,neuron][start_index-1100:start_index]
        _, wavelet_scalogram, _ = morlet_wavelet_transform(batch, 1000)
        neuron_data.append(wavelet_scalogram.flatten())
    time_data = np.hstack(neuron_data)
    del neuron_data
    input_dataset.append(time_data)

input_df = np.vstack(input_dataset)
input_df = pd.DataFrame(input_df)
input_df.to_csv('data/preprocessed_input.csv', index=False)
'''





inputs = pd.read_csv('data/preprocessed_input.csv')
targets = pd.read_csv('data/targets.csv')


X = inputs.values
Y = targets.values[:, 1:]
val_idx = np.argmax(targets.values[:, 0] >= 600)

X_train = X[:val_idx, :]
y_train = Y[:val_idx, :]
X_dev = X[val_idx:, :]
y_dev = Y[val_idx:, :]



print('PLS REGRESSION')
pls = PLSRegression(n_components = 21)
pls.fit(X_train, y_train)



Y_pred = pls.predict(X_train)
t_corr = [np.corrcoef(y_train[:, j], Y_pred[:, j])[0, 1] for j in range(Y.shape[1])]
t_r2 = [r2_score(y_train[:, j], Y_pred[:, j]) for j in range(Y.shape[1])]
    


Y_pred = pls.predict(X_dev)
v_corr = [np.corrcoef(y_dev[:, j], Y_pred[:, j])[0, 1] for j in range(Y.shape[1])]
v_r2 = [r2_score(y_dev[:, j], Y_pred[:, j]) for j in range(Y.shape[1])]

print(np.mean(Y_pred, axis=0))


print(f'training $R^2$ = {t_r2}')
print(f'training correlation coef = {t_corr}')

print(f'dev $R^2$ = {v_r2}')
print(f'dev correlation coef = {v_corr}\n\n\n')
np.savetxt('output/PLS_pred.csv', Y_pred, delimiter=',')




print('XGBOOST REGRESSOR')
xgboost_seed = 19
learning_rate = 0.1
max_depth = 5
xgb_r = xg.XGBRegressor(objective ='reg:squarederror', max_depth=max_depth,
                        learning_rate = learning_rate, reg_alpha = 15,
                        reg_lambda = 10, seed = xgboost_seed, n_estimators = 100)

# Fitting the model
xgb_r.fit(X_train, y_train)


Y_pred = xgb_r.predict(X_train)
t_corr = [np.corrcoef(y_train[:, j], Y_pred[:, j])[0, 1] for j in range(Y.shape[1])]
t_r2 = [r2_score(y_train[:, j], Y_pred[:, j]) for j in range(Y.shape[1])]


Y_pred = xgb_r.predict(X_dev)

np.savetxt('output/XGB_pred.csv', Y_pred, delimiter=',')
v_corr = [np.corrcoef(y_dev[:, j], Y_pred[:, j])[0, 1] for j in range(Y.shape[1])]
v_r2 = [r2_score(y_dev[:, j], Y_pred[:, j]) for j in range(Y.shape[1])]

print(f'training $R^2$ = {t_r2}')
print(f'training correlation coef = {t_corr}')

print(f'dev $R^2$ = {v_r2}')
print(f'dev correlation coef = {v_corr}\n\n\n')



'''
# Define the hyperparameter grid for a sweep
param_grid = {
    'max_depth': [5],
    'learning_rate': [0.1],
    'lambda' : [0, 0.01, 0.1, 1, 10, 15], # l2 penalty
    'alpha' : [0, 0.01, 0.1, 1, 10, 15] # l1 penalty
}


# Create the GridSearchCV object
#grid_search = GridSearchCV(xgb_r, param_grid, cv=5, verbose=3, scoring=test)
grid_search = GridSearchCV(xgb_r, param_grid, cv=5, verbose=3, scoring='neg_mean_squared_error')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best set of hyperparameters and the corresponding score
print("Best set of hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
'''