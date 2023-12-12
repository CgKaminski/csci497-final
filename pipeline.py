from dataset import K1Dataset, load_data
from preprocessing import *
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


fs = 1000 # sampling freq (Hz)
ecog_data, motion_data = load_data()

batch = ecog_data.head(1100)

for key in tqdm(batch.keys()[1:], desc='Band Pass Filter'):
    batch[key] = band_pass_filter(batch[key])

filt_ecog_data = batch.values[:,1:]
car_ecog_data = car(filt_ecog_data)
downsampled_hand_data = downsample(motion_data.values[:,-3:])


for start_time in time - 1100:
    for n in neurons:
        batch = ecog_data[n][start_time:start_time+1100]
        wavelet_scalogram = morlet_wavelet_transform(ecog_data['ECoG_ch1'].values, 1000)

breakpoint()
