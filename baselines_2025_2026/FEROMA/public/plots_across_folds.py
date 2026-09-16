# read data json
import json
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import config as cfg
from matplotlib import rcParams



###################################################################################################
# ARGUMENTS
###################################################################################################
parser = argparse.ArgumentParser(description='Plot accuracy and derivative across folds')
parser.add_argument('--show', default=False, help='Show the plots', choices=[True, False])
parser.add_argument('--dataset', default='CIFAR10', help='Dataset name', choices=['CIFAR10', 'CIFAR100', 'MNIST', 'FMNIST'])
args = parser.parse_args()


###################################################################################################
# READ DATA
###################################################################################################
data = []
for i in range(cfg.k_folds):
    with open(f'{cfg.strategy}/histories/{cfg.default_path}/distributed_metrics_{i}.json') as f:
        data.append(json.load(f))



###################################################################################################
# CONFIDENCE INTERVALS AND DERIVATIVES - TRAINING ACCURACY
###################################################################################################
# Extract accuracy lists from each run
accuracy_runs = np.array([x['accuracy'] for x in data])

# Number of runs and steps
num_runs, num_steps = accuracy_runs.shape

# Compute mean accuracy per step
mean_accuracy = np.mean(accuracy_runs, axis=0)

# Compute standard error per step
sem_accuracy = stats.sem(accuracy_runs, axis=0)

# Compute 95% confidence intervals
confidence_level = 0.95
degrees_freedom = num_runs - 1
confidence_intervals = sem_accuracy * stats.t.ppf((1 + confidence_level) / 2., degrees_freedom)

# Compute derivatives for each run
derivatives = np.diff(accuracy_runs, axis=1) 

# Compute mean derivative per step
mean_derivative = np.mean(derivatives, axis=0)

# Compute standard error for derivatives
sem_derivative = stats.sem(derivatives, axis=0)

# Compute 95% confidence intervals for derivatives
confidence_intervals_derivative = sem_derivative * stats.t.ppf((1 + confidence_level) / 2., degrees_freedom)

# Define step indices
steps_accuracy = np.arange(1, num_steps + 1)
steps_derivative = np.arange(2, num_steps + 1)  

# ICML plot settings
def setup_icml_plot(two_column=False):
    """Set up ICML-compatible plot settings."""
    if two_column:
        figure_width = 7  # Full-page width for two-column layout (in inches)
    else:
        figure_width = 3.5  # Half-page width for two-column layout (in inches)
 
    rcParams.update({
        # Font and text
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Use serif fonts
        "font.serif": ["Times New Roman"],  # Set font to Times New Roman

        'font.size': 14,             # Base font size
        'figure.figsize': (18, 8),   # Increased width for side-by-side plots
        'axes.labelsize': 16,        # Axis labels font size
        'axes.titlesize': 19,        # Titles font size
        'legend.fontsize': 14,       # Legend font size
        'xtick.labelsize': 14,       # X-ticks font size
        'ytick.labelsize': 14,        # Y-ticks font size
 
        # Line and marker styles
        "lines.linewidth": 1.2,  # Line width
        "lines.markersize": 3,  # Marker size
 
        # Figure dimensions
        "figure.figsize": (figure_width, figure_width * 0.15),  # TODO change to better ratio
        "figure.dpi": 300,  # High resolution for publication
 
        # Grid
        "axes.grid": True,  # Enable grid
        "grid.alpha": 0.3,  # Grid transparency
        "grid.linestyle": "--",  # Dashed grid lines
 
        # Legend
        "legend.frameon": False,  # No border around legends
    })

# Create a figure with two subplots
plt.figure(figsize=(7, 9))
setup_icml_plot(two_column=True)

# ---- Top Plot: Accuracy ----
plt.subplot(2, 1, 1)
plt.plot(steps_accuracy, mean_accuracy, marker='o', linestyle='-', color='blue', label='Mean Accuracy')
plt.fill_between(steps_accuracy,
                 mean_accuracy - confidence_intervals,
                 mean_accuracy + confidence_intervals,
                 color='blue', alpha=0.2, label='95\\% Confidence Interval')
plt.title(f'Mean Accuracy Over Rounds - {args.dataset}')
plt.xlabel('Training Rounds')
plt.ylabel('Accuracy')
plt.xticks(steps_accuracy)
plt.legend()
plt.grid(True)

# ---- Bottom Plot: Derivative of Accuracy ----
plt.subplot(2, 1, 2)
plt.plot(steps_derivative, mean_derivative, marker='x', linestyle='-', color='red', label='Mean Derivative')
plt.fill_between(steps_derivative,
                 mean_derivative - confidence_intervals_derivative,
                 mean_derivative + confidence_intervals_derivative,
                 color='red', alpha=0.2, label='95\\% Confidence Interval')
plt.title(f'Mean Derivative of Accuracy - {args.dataset}')
plt.xlabel('Training Rounds')
plt.ylabel('Change in Accuracy')
plt.xticks(steps_derivative)
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.show()
plt.savefig(f'{cfg.strategy}/images/{cfg.default_path}/accuracy_derivative_{cfg.non_iid_type}.pdf')
if args.show:
    plt.show()
else:
    plt.close()
