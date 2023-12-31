from dataset import K1Dataset, load_data
from preprocessing import *
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt



#inputs = pd.read_csv('data/inputs.csv', delimiter='|')
#outputs = pd.read_csv('data/outputs.csv', delimiter='|')



#breakpoint()

fs = 1000 # sampling freq (Hz)
ecog_data, motion_data = load_data()

for key in tqdm(ecog_data.keys()[1:], desc='Band Pass Filter'):
    ecog_data[key] = band_pass_filter(ecog_data[key].values)

filt_ecog_data = ecog_data.values[:,1:]
#filt_ecog_data = car(filt_ecog_data)

hand_data, time = downsample(motion_data[['MotionTime','Experimenter:RHNDx','Experimenter:RHNDy','Experimenter:RHNDz']])


#hand_df = pd.DataFrame(np.concatenate((time.reshape(-1,1), hand_data), axis=1), columns=['Time', 'Hand:x', 'Hand:y', 'Hand:z'])
hand_df = pd.DataFrame(hand_data, columns=['Time', 'Hand:x', 'Hand:y', 'Hand:z'])
hand_df = hand_df[hand_df['Time'] > 1.1]
hand_df.to_csv('data/targets.csv')

input_dataset = []
for start_time in tqdm(hand_df['Time'].values, desc='Wavelet Transform for all time steps'):
    neuron_data = []
    for neuron in range(64):
        start_index = ecog_data.index[ecog_data['ECoG_time'] == start_time][0]
        batch = filt_ecog_data[:,neuron][start_index-1100:start_index]
        batch = np.reshape(batch, (10, 10, -1))
        batch = batch.mean(axis=2)

        neuron_data.append(batch.flatten())
    time_data = np.hstack(neuron_data)
    del neuron_data
    input_dataset.append(time_data)

input_df = np.vstack(input_dataset)
input_df = pd.DataFrame(input_df)
input_df.to_csv('data/preprocessed_input.csv')