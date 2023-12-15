import numpy as np
import matplotlib.pyplot as plt

def plot_xgb_scores(xgb_data: np.ndarray[..., 3]) -> None:
    """
    Plots the correlation coefficients of the XGBoost model for different numbers of estimators.
    :param xgb_data: A numpy array containing X, Y, Z correlation coefficients for different numbers of estimators.
    :return: None
    """
    num_dimensions = 3
    num_estimators = xgb_data.shape[0]

    # Set the figure size
    fig, ax = plt.subplots(figsize=(18, 9))

    # Plot the results for each dimension
    for dim in range(num_dimensions):
        ax.plot(range(1, num_estimators + 1), xgb_data[:, dim], label=f'{["X", "Y", "Z"][dim]}-position')

    ax.set_xlabel('Number of Boosted Trees', fontsize=14)
    ax.set_ylabel('Correlation Coefficient', fontsize=14)
    ax.legend(fontsize=12)

    # Adjust x-axis tick size
    ax.tick_params(axis='x', which='major', labelsize=12)

    plt.savefig('XGB_N_est.png', bbox_inches='tight')  # Add bbox_inches='tight' to remove extra white space

if __name__ == '__main__':
    # Load data
    xgb_corr_data = np.genfromtxt('../output/XGB_N_est.csv', delimiter=',', skip_header=1)
    plot_xgb_scores(xgb_corr_data)
