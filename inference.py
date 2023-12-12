from dataset import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import band_pass_filter, car, morlet_wavelet_transform
from scipy.signal import butter, sosfilt, morlet2, spectrogram
import pywt


def scalogram_figs(data, save=False):

    scalogram, scalogram_bin, normalized_scalogram = morlet_wavelet_transform(data, 1000)

    # Create subplots
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Plot data on each subplot
    axs[0].plot(np.arange(len(data)), data)
    axs[0].set_xlabel('Time (t-1.1s - t)')
    axs[0].set_ylabel('Voltage (uV)')
    axs[0].set_title('ECoG Signal')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(scalogram, aspect='auto')
    axs[1].set_title('Scalogram')
    axs[1].set_xlabel('Time (t-1.1s - t)')
    axs[1].set_ylabel('Frequency (Hz)')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    axs[2].imshow(scalogram_bin)
    axs[2].set_title('10 by 10 Scalogram Matrix')
    axs[2].set_xlabel('Time  (t-1.1s - t)(t-1.1s - t)')
    axs[2].set_ylabel('Frequency (Hz)')
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    axs[3].imshow(normalized_scalogram)
    axs[3].set_title('Normalized Scalogram Matrix')
    axs[3].set_xlabel('Time (t-1.1s - t)')
    axs[3].set_ylabel('Frequency (Hz)')
    axs[3].set_xticks([])
    axs[3].set_yticks([])

    plt.tight_layout()

    if save: 
        plt.savefig('docs/figs/scalogram_figs.png')
    else: 
        plt.show()


def main():
    ecog, _ = load_data()
    
    scalogram_figs(ecog.head(1100)['ECoG_ch1'], save=True)


if __name__ == '__main__':
    main()
