import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def calculate_correlation_coefficients(true_data: np.ndarray[..., 3], pred_data: np.ndarray[..., 3]) -> list[float]:
    correlation_coefficients = [np.corrcoef(true_data[:, dim], pred_data[:, dim])[0, 1] for dim in range(3)]
    return correlation_coefficients


def plot_coordinates(true_data: np.ndarray[..., 3], pred_data: np.ndarray[..., 3], total_time: int = 5,
                     file_name: str = "plot") -> None:
    """
    Plots the coordinates of the true and predicted trajectories in 3 sub-plots (one for each dimension).
    :param true_data: The true trajectory data.
    :param pred_data: The predicted trajectory data.
    :param total_time: The total time of the trajectory, (assume 5min).
    :param file_name: The name of the file to save the plot to.
    :return: None.
    """
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
                          va='center')

        axs[dim].set_xlabel('Time (min)')

    # Set y-axis label for each coordinate
    for dim, title in enumerate(['X-position (normalized)', 'Y-position (normalized)', 'Z-position (normalized)']):
        axs[dim].set_ylabel(title)

    axs[0].legend(bbox_to_anchor=(1.0, 1.25), loc="upper right", frameon=False, ncol=1)

    plt.tight_layout()
    plt.savefig(file_name)


if __name__ == '__main__':
    # Load pred. data
    pls_data = np.genfromtxt('../output/PLS_pred.csv', delimiter=',', skip_header=1)
    xgb_data = np.genfromtxt('../output/XGB_pred.csv', delimiter=',', skip_header=1)

    # Load motion data, extract last 3 cols. (x, y, z)
    motion_data = np.genfromtxt('../data/targets.csv', delimiter=',', skip_header=1)[-pls_data.shape[0]:, -3:]

    # Plot coordinates
    plot_coordinates(motion_data, pls_data, file_name='PLS_plot')
    plot_coordinates(motion_data, xgb_data, file_name='XGB_plot')
