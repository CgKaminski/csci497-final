from scipy.signal import butter, sosfilt, morlet2, spectrogram
from tqdm import tqdm
import pandas
import pywt
import numpy as np
import matplotlib.pyplot as plt

def main():
    ecog_path = 'data/ECoG.csv'
    motion_path = 'data/Motion.csv'
    ecog_data = pandas.read_csv(ecog_path)
    motion_data = pandas.read_csv(motion_path)

    # print(ecog_data)
    # print(motion_data)

    
    # band pass filter from 0.1 to 600 Hz
    for key in ecog_data.keys():
        if key != 'ECoG_time':
            ecog_data[key] = band_pass_filter(ecog_data[key])


    # re-reference using CAR montage
    # downsample motion markers to 20 Hz
    # Morlet DWT at 10 different freqs 10-150Hz in logarithmic scale
    # Scalogram at t calculated from t-1.1s to t
    # sample scalogram at 10 time lags (t-100ms, t-200ms, ..., t-1s) to form 10x10 scalogram matrix of time t
    # normalize scalogram by calcilating z-score at each freq bin
    # pooling ? to form scalo vector
    # PLS
    # form M
    
    return

def band_pass_filter(data):
    fs = 1000  # Sampling rate in Hz
    low = 0.1  # Low cutoff frequency in Hz
    high = 499.9999  # High cutoff frequency in Hz

    N = 3  # Filter order
    sos = butter(N, [low, high], btype='bandpass', output='sos', fs=fs)  # Build filter

    batches = int(len(data) / fs)
    filtered = []

    for i in range(batches):
        # Apply filter over each second
        batch_data = data[i * fs:(i + 1) * fs]
        filtered_batch = sosfilt(sos, batch_data)
        filtered.extend(filtered_batch)

    # Filter remaining data if any
    remaining_data = data[batches * fs:]
    if remaining_data.size > 0:
        filtered.extend(sosfilt(sos, remaining_data))

    return np.array(filtered)

'''
def car(data):
    avg_reference = data - np.mean(data, axis=0)
    return avg_reference
'''
def car(ecog_data):
    average_signal = np.mean(ecog_data, axis=1)
    car_ecog_data = ecog_data - average_signal[:, np.newaxis]
    return car_ecog_data

def downsample(df , original_fs=1000, target_fs=20):
    data = df.values
    times = df['MotionTime'].values
    factor = int(original_fs / target_fs)
    downsampled_data = data[::factor]
    downsampled_time = times[::factor]
    return downsampled_data, downsampled_time

def morlet_wavelet_transform(batch, fs):
    ''' input - ecog data in time window for a single neuron
        output - scalogram, scalogram_bin, normalized_scalogram  '''

    center_freqs = np.logspace(np.log10(10), np.log10(150), 10)
    num_freqs = 10
    coarsest_scale = 7
    time_window = int(1.1 * fs)

    scalograms = []
    
    scalogram_t, _ = pywt.cwt(batch, center_freqs, 'morl')
    scalogram = np.abs(scalogram_t)
    
    scalogram_bin = scalogram.reshape(10, 10, -1)
    scalogram_bin = scalogram_bin.mean(axis=2)

    mean = scalogram_bin.mean(axis=1)
    std = scalogram_bin.std(axis=1)

    normalized_scalogram = (scalogram_bin - mean[:, np.newaxis]) / std[:, np.newaxis]

    return scalogram, scalogram_bin, normalized_scalogram 
    


if __name__ == '__main__':
    main()
