import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams


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
        
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 19,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
 
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

# Data for MNIST
mnist_means = [0.9190, 0.9228, 0.9453, 0.9471, 0.9495, 0.9468, 0.9542, 0.9515, 0.9445]
mnist_stds = [0.026666667, 0.026666667, 0.016666667, 0.02, 0.02, 
              0.023333333, 0.01, 0.01, 0.016666667] / np.sqrt(5)

# Data for FMNIST
fmnist_means = [0.8574, 0.8583, 0.8620, 0.8623, 0.8614, 0.8595, 0.8615, 0.8608, 0.8595]
fmnist_stds = [0.0200, 0.0200, 0.0133, 0.0167, 0.0167, 
               0.0167, 0.0167, 0.0167, 0.0167] / np.sqrt(5)

# Data for CIFAR-10
cifar_means = [0.5142, 0.5238, 0.5248, 0.5227, 0.5231, 0.5215, 0.5192, 0.5188, 0.5187]
cifar_stds = [0.02, 0.02, 0.02, 0.016666667, 0.016666667, 
             0.016666667, 0.016666667, 0.02, 0.013333333] / np.sqrt(5)



# X-axis labels (e.g., different experiments or models)
x_labels = [f"{i+1}" for i in range(len(mnist_means))]

# Create subplots
setup_icml_plot(two_column=True)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

# Plot for MNIST
axes[0].errorbar(x_labels, mnist_means, yerr=mnist_stds, fmt='o-', 
                ecolor='b', capsize=5, capthick=1, markerfacecolor='blue', markersize=5, 
                linestyle='-', color='blue', label='MNIST Accuracy')
axes[0].set_title('Model Performance on MNIST')
axes[0].set_xlabel('Clustering Round')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0.90, 0.96)
# axes[0].legend()

# Plot for FMNIST
axes[1].errorbar(x_labels, fmnist_means, yerr=fmnist_stds, fmt='o-', 
                ecolor='r', capsize=5, capthick=1, markerfacecolor='red', markersize=5, 
                linestyle='-', color='red', label='FMNIST Accuracy')
axes[1].set_title('Model Performance on FMNIST')
axes[1].set_xlabel('Clustering Round')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0.845, 0.875)

# Plot for CIFAR-10
axes[2].errorbar(x_labels, cifar_means, yerr=cifar_stds, fmt='o-', 
                ecolor='g', capsize=5, capthick=1, markerfacecolor='green', markersize=5, 
                linestyle='-', color='green', label='CIFAR-10 Accuracy')
axes[2].set_title('Model Performance on CIFAR-10')
axes[2].set_xlabel('Clustering Round')
axes[2].set_ylabel('Accuracy')
axes[2].set_ylim(0.50, 0.54)
# axes[2].legend()

# Improve layout
plt.tight_layout()

# Save the figure (optional)
# plt.subplots_adjust(wspace=0.25)  # Increase wspace as needed (default is ~0.2)
plt.savefig('stop_round.pdf', dpi=300)

# Display the plot
plt.show()







# Second plot for threshold detection
# Data for MNIST
mnist_means = [0.9404, 0.9405, 0.9381, 0.9378, 0.9375, 0.9368]
mnist_stds = [0.0143, 0.0140, 0.0140, 0.0148, 0.0141, 0.0140] / np.sqrt(5)

# Data for CIFA-10
cifar_means = [0.4270, 0.4271, 0.4272, 0.4260, 0.4253, 0.4243]
cifar_stds = [0.0194, 0.0193, 0.0197, 0.0204, 0.0205, 0.0204] / np.sqrt(5)

# X-axis labels (e.g., different experiments or models)
x_labels = [0.8, 0.6, 0.4, 0.2, 0.1, 0.005]

# Create subplots
setup_icml_plot(two_column=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

# Plot for MNIST
axes[0].errorbar(x_labels, mnist_means, yerr=mnist_stds, fmt='o-', 
                ecolor='b', capsize=5, capthick=1, markerfacecolor='blue', markersize=5, 
                linestyle='-', color='blue', label='MNIST Accuracy')
axes[0].set_title('Model Performance on MNIST')
axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0.93, 0.95)
# axes[0].legend()

# Plot for FMNIST
axes[1].errorbar(x_labels, cifar_means, yerr=cifar_stds, fmt='o-', 
                ecolor='r', capsize=5, capthick=1, markerfacecolor='red', markersize=5, 
                linestyle='-', color='red', label='CIFAR-10 Accuracy')
axes[1].set_title('Model Performance on CIFAR-10')
axes[1].set_xlabel('Threshold')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0.41, 0.44)

# Improve layout
plt.tight_layout()

# Save the figure (optional)
plt.subplots_adjust(wspace=0.25)  # Increase wspace as needed (default is ~0.2)
plt.savefig('threshold_detection.pdf', dpi=300)

# Display the plot
plt.show()
