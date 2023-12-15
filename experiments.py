from dataset import load_data
from preprocessing import *
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
import xgboost as xg
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


# generate data if not loading from a file



def gen_data():
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




def sweep_n_components(X_train, y_train, X_dev, y_dev):
    x_vals = []
    PRESS_vals = []
    r2_vals = []
    max_n=100
    print(f"Sweep over {max_n} components for PLS")
    for num in range(max_n):
        #   PRESS = []
        print(f'N = {num+1} of {max_n}')
        #for i in range(n_trials):
        
        pls = PLSRegression(n_components = num + 1)
        pls.fit(X_train, y_train)
        
        Y_pred = pls.predict(X_dev)
        
        r2 = [r2_score(y_dev[j, :], Y_pred[j, :]) for j in range(Y_pred.shape[0])]
        
        PRESS_vals += list(np.sum((y_dev - Y_pred)**2, axis=1))   
        r2_vals += r2
        x_vals += [(num + 1)] * len(r2)
        #print(len(PRESS))
        #print(len(r2))
        
        #PRESS_error.append(PRESS)
        #    PRESS_vals.append(np.mean(PRESS))
        #r2_error.append(np.std(r2))
        #r2_vals.append(np.mean(r2))
        
    result_df = {
        'domain' : x_vals,
        'PRESS_mean' : PRESS_vals,
        #'PRESS_err' : PRESS_error,
        'R2_mean' : r2_vals
        #'R2_err' : r2_error
    }

    result_df = pd.DataFrame(result_df)
    result_df.to_csv('output/PLS_N_components.csv')
    #print(result_df)



# PLS fitting
def fit_PLS(X_train, y_train, X_dev, y_dev):
    print('PLS REGRESSION')
    pls = PLSRegression(n_components = 4)
    pls.fit(X_train, y_train)
    
    
    
    Y_pred = pls.predict(X_train)
    t_corr = [np.corrcoef(y_train[:, j], Y_pred[:, j])[0, 1] for j in range(Y.shape[1])]
    t_r2 = [r2_score(y_train[:, j], Y_pred[:, j]) for j in range(Y.shape[1])]
    
    
    
    Y_pred = pls.predict(X_dev)
    v_corr = [np.corrcoef(y_dev[:, j], Y_pred[:, j])[0, 1] for j in range(Y.shape[1])]
    v_r2 = [r2_score(y_dev[:, j], Y_pred[:, j]) for j in range(Y.shape[1])]
    
    
    print(f'training $R^2$ = {t_r2}')
    print(f'training correlation coef = {t_corr}')
    
    print(f'dev $R^2$ = {v_r2}')
    print(f'dev correlation coef = {v_corr}\n\n\n')
    np.savetxt('output/PLS_pred.csv', Y_pred, delimiter=',')




# XGBoost fitting
def fit_XGBoost(X_train, y_train, X_dev, y_dev):
    print('XGBOOST REGRESSOR')
    xgboost_seed = 19
    learning_rate = 0.1
    max_depth = 5
    reg_alpha = 1
    reg_lambda = 0.01
    n_est = 10
    xgb_r = xg.XGBRegressor(objective ='reg:squarederror', max_depth=max_depth,
                            learning_rate = learning_rate, reg_alpha = reg_alpha,
                            reg_lambda=reg_lambda, seed=xgboost_seed, n_estimators=n_est)

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


 
def sweep_n_est(X_train, y_train, X_dev, y_dev):
    # XGBoost n_estimators sweep
    
    
    xgboost_seed = 19
    
    learning_rate = 0.1
    max_depth = 5
    reg_alpha = 1
    reg_lambda = 0.01
    corr = []
    n_est = 30
    n_estimators = range(n_est)
    print(f"Sweep over {n_est} esimators for XGBoost")
    for n in n_estimators:
        print(f'n = {n + 1}')
        xgb_r = xg.XGBRegressor(objective ='reg:squarederror', max_depth=max_depth,
                            learning_rate = learning_rate, reg_alpha = reg_alpha,
                            reg_lambda=reg_lambda, seed=xgboost_seed, n_estimators=n+1)

        # Fitting the model
        xgb_r.fit(X_train, y_train)
        
        
        Y_pred = xgb_r.predict(X_train)
        t_corr = [np.corrcoef(y_train[:, j], Y_pred[:, j])[0, 1] for j in range(Y.shape[1])]
        t_r2 = [r2_score(y_train[:, j], Y_pred[:, j]) for j in range(Y.shape[1])]
        
        
        Y_pred = xgb_r.predict(X_dev)
        
        #np.savetxt('output/XGB_pred.csv', Y_pred, delimiter=',')
        v_corr = [np.corrcoef(y_dev[:, j], Y_pred[:, j])[0, 1] for j in range(Y.shape[1])]
        v_r2 = [r2_score(y_dev[:, j], Y_pred[:, j]) for j in range(Y.shape[1])]
        
        #print(f'training $R^2$ = {t_r2}')

        #print(f'training correlation coef = {t_corr}')
        
        #print(f'dev $R^2$ = {v_r2}')
        #print(f'dev correlation coef = {v_corr}\n\n\n')
        corr.append(v_corr)
    np.savetxt('output/XGB_N_est.csv', corr, delimiter=',')




# XGBoost Grid Search

def XGB_grid_search(X_train, y_train, X_dev, y_dev):
    # Define the hyperparameter grid for a sweep
    param_grid = {
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 1],
        'reg_lambda' : [0, 0.01, 0.1, 1, 10], # l2 penalty
        'reg_alpha' : [0, 0.01, 0.1, 1, 10] # l1 penalty
    }
    
    xgb_r = xg.XGBRegressor(objective ='reg:squarederror', n_estimators=10)
    # Create the GridSearchCV object
    grid_search = GridSearchCV(xgb_r, param_grid, cv=5, verbose=3, scoring='neg_root_mean_squared_error')
    
    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)
    
    # Print the best set of hyperparameters and the corresponding score
    print("Best set of hyperparameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)


if __name__ == "__main__":
#    gen_data()
    inputs = pd.read_csv('data/preprocessed_input.csv')
    targets = pd.read_csv('data/targets.csv')
    X = inputs.values
    Y = targets.values[:, 1:]
    val_idx = np.argmax(targets.values[:, 0] >= 600)
    
    X_train = X[:val_idx, :]
    y_train = Y[:val_idx, :]
    X_dev = X[val_idx:, :]
    y_dev = Y[val_idx:, :]
    fit_PLS(X_train, y_train, X_dev, y_dev)
    fit_XGBoost(X_train, y_train, X_dev, y_dev)
    sweep_n_components(X_train, y_train, X_dev, y_dev)
    sweep_n_est(X_train, y_train, X_dev, y_dev)
    # XGB_grid_search(X_train, y_train, X_dev, y_dev)    
