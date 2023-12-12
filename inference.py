from dataset import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ecog_signal(data, save=False):
    plt.plot(np.arange(len(data)), data)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (uV)')
    plt.title('ECoG Signal')
    if save: 
        plt.savefig('docs/figs/ecog_signal.png')
    else: 
        plt.show()

def main():
    ecog, _ = load_data()
    
    ecog_signal(ecog.head(1100)['ECoG_ch63'], save=False)


if __name__ == '__main__':
    main()
