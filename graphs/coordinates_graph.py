import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.stats import pearsonr

def calculate_correlation_coefficients(true_data: np.ndarray, pred_data: np.ndarray) -> list:
    correlation_coefficients = [np.corrcoef(true_data[:, dim], pred_data[:, dim])[0, 1] for dim in range(3)]
    return correlation_coefficients


def plot_coordinates(true_data: np.ndarray, pred_data: np.ndarray, total_time: int = 5,
                     file_name: str = "plot") -> None:
    """
    Plots the coordinates of the true and predicted trajectories in 3 sub-plots (one for each dimension).
    :param true_data: The true trajectory data.
    :param pred_data: The predicted trajectory data.
    :param total_time: The total time of the trajectory, (assume 5min).
    :param file_name: The name of the file to save the plot to.
    :return: None.
    """
    plt.clf()
    num_dimensions = 3

    fig, axs = plt.subplots(num_dimensions, 1, figsize=(6 * num_dimensions, 3 * num_dimensions))

    # Calc. time values for time interval (we assume 5min)
    time_values = np.linspace(0, total_time, len(true_data))

    for i, (data, label, color) in enumerate(zip([true_data, pred_data], ['Observed', 'Predicted'], ['blue', 'red'])):
        for dim in range(num_dimensions):
            # Plot each dimension separately in mpl sub-plot
            axs[dim].plot(time_values, data[:, dim], label=f'{label} trajectories', color=color, linewidth=1)

            # Calculate R-squared score
            correlation_coefficients = calculate_correlation_coefficients(true_data, pred_data)

            # Annotate the subplot with R-squared score
            axs[dim].text(0.9, 0.05, f'r : {correlation_coefficients[dim]:.2f}', transform=axs[dim].transAxes, ha='center',
                          va='center', fontsize=14)

            # Set tick label size
            axs[dim].tick_params(axis='both', which='major', labelsize=14)

        axs[dim].set_xlabel('Time (min)', fontsize=14)

    # Set y-axis label for each coordinate
    for dim, title in enumerate(['X-position (normalized)', 'Y-position (normalized)', 'Z-position (normalized)']):
        axs[dim].set_ylabel(title, fontsize=14)

    axs[0].legend(bbox_to_anchor=(1.0, 1.35), loc="upper right", frameon=False, fontsize=14, ncol=1)

    plt.tight_layout()
    plt.savefig(file_name)

def plot_n_est(data, filename):
    plt.clf()
    plt_1 = plt.figure(figsize=(10, 4))
    x = range(1, len(data) +1)
    plt.plot(x, data.T[0], label='X')
    plt.plot(x, data.T[1], label='Y')
    plt.plot(x, data.T[2], label='Z')
    plt.legend()
    plt.xlabel('Number of boosted trees')
    plt.ylabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(filename)

    
def plot_n_components(data, filename):
    plt.clf()
    plt_1 = plt.figure(figsize=(10, 4))
    #    plt.errorbar(data['domain'], data['PRESS_mean'], data['PRESS_err'], label='PRESS')
    #    plt.errorbar(data['domain'], data['R2_mean'], data['R2_err'], label='R2')
    plt.ylabel(r'$R^2$')
    plt.xlabel('Number of PLS Components')
        
    sns.lineplot(data=data, x= 'domain', y = 'R2_mean', color='green', errorbar='sd')

    ax2 = plt.twinx()
    

    plt.ylabel('PRESS')
    sns.lineplot(data=data, x='domain', y='PRESS_mean', color='blue', ax=ax2, errorbar='sd')

    #    plt.tight_layout()
    plt.savefig(filename)
    
if __name__ == '__main__':
    # Load pred. data
    pls_data = np.genfromtxt('../output/PLS_pred.csv', delimiter=',', skip_header=1)
    xgb_data = np.genfromtxt('../output/XGB_pred.csv', delimiter=',', skip_header=1)


    xgb_est_data = np.genfromtxt('../output/XGB_N_est.csv', delimiter=',')
    pls_n_data = pd.read_csv('../output/PLS_N_components.csv')

    # Load motion data, extract last 3 cols. (x, y, z)
    motion_data = np.genfromtxt('../data/targets.csv', delimiter=',', skip_header=1)[-pls_data.shape[0]:, -3:]

    # Plot coordinates
    plot_coordinates(motion_data, pls_data, file_name='PLS_plot')
    plot_coordinates(motion_data, xgb_data, file_name='XGB_plot')
    plot_n_est(xgb_est_data, 'XGB_N_est')
    plot_n_components(pls_n_data, 'PLS_N_components')