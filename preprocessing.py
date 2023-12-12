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
    # pooling  to form scalo vector
    # PLS
    # form M
    
    return


def band_pass_filter(data):
    fs = 1000 # 1 kHz == 1000Hz sampling rate
    nyq = 0.5 * fs # calculate nyquist 
    low = 0.1 / nyq # 0.1 Hz low end of filter
    high = 400 / nyq # should be 600 Hz high end of filter -- had to use 400 to get code working.

    N = 3 # order
    Wn = [low, high] # critical frequencies 
    sos = butter(N, Wn, btype='bandpass', output='sos') # build filter

    
    batches = int(len(data) / fs)
    filtered = []

    for i in range(batches):
        # apply filter over each second
        batch_data = data[:fs]
        data = data[fs:]
        filtered += list(sosfilt(sos, batch_data)) 

    # filter left-over data
    filtered += list(sosfilt(sos, data))

    return filtered

def car(data):
    avg_reference = data - np.mean(data, axis=0)
    return avg_reference

def downsample(df , original_fs=1000, target_fs=20):
    data = df.values
    times = df['MotionTime'].values
    factor = int(original_fs / target_fs)
    downsampled_data = data[::factor]
    downsampled_time = times[::factor]
    return downsampled_data, downsampled_time

# perform morlet
def morlet_wavelet_transform(batch, fs):
    '''
    Performs Morlet Wavelet Transform on a batch and forms a scalogram.
    
    Input:
       batch : ecog data in time window for a single neuron
       fs : sampling frequency
    
    Output:
       scalogram: original scalogram after wavelet transform
       scalogram_bin : scalogram reshaped to 10x10
       normalized_scalogram: normalized reshaped scalogram
    '''

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
