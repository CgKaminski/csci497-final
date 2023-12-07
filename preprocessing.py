import scipy.signal import butter, sosfilt

def main():
    # band pass filter from 0.1 to 600 Hz
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
    order = 1
    Wn = [0.1, 600]
    sos = butter(order, Wn, output='sos')
    filtered = sosfilt(sos, data)
    return filtered

if __name__ == '__main__':
    main()