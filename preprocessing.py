from scipy.signal import butter, sosfilt, morlet2, spectrogram
from neurodsp.timefrequency import compute_wavelet_transform
from tqdm import tqdm
import pandas
import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def band_pass_filter(data, lowcut=0.1, highcut=499, fs=1000, order=3):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

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
    center_freqs = np.logspace(np.log10(10), np.log10(150), 10)

    scalogram_t = compute_wavelet_transform(batch, fs=fs, freqs=center_freqs)
    scalogram = np.abs(scalogram_t)

    scalogram_bin = scalogram.reshape(10, 10, -1)
    scalogram_bin = scalogram_bin.mean(axis=2)

    mean = scalogram_bin.mean(axis=0)
    std = scalogram_bin.std(axis=0)

    normalized_scalogram = (scalogram_bin - mean[np.newaxis, :]) / std[np.newaxis, :]

    return scalogram, scalogram_bin, normalized_scalogram


if __name__ == '__main__':
    main()
