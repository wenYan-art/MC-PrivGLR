import matplotlib.pyplot as plt
import numpy as np
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
        "axes.labelsize": 9,  # Font size for axis labels
        "axes.titlesize": 9,  # Font size for titles
        "legend.fontsize": 4,  # Font size for legends
        "xtick.labelsize": 8,  # Font size for x-axis ticks
        "ytick.labelsize": 8,  # Font size for y-axis ticks
 
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
 
# LOADING DATA =================================================================
# ==============================================================================
x = np.linspace(0, 4, 4)  # Generate 4 equally spaced x values
 
# Data_load_no_inference
# zzz_no_p = [0.928501334, 0.867970587, 0.817063782, 0.840793491]
# zzz_no = [0.928501334,0.867970587 , 0.817063782, 0.840793491]  
# fedavg_no = [0.844397593,0.700330634 , 0.661952839, 0.658734032]
# ifca_no = [0.809057795,0.760109587 , 0.762366395, 0.744451775]
# fedrc_no = [0.747058886,0.310470395 , 0.283623699, 0.285470752]
# fedem_no = [0.75658316,0.326672057 , 0.292381201, 0.299923074]
# fesem_no = [0.826859755,0.754151627 , 0.72082539, 0.684889664]
# feddrift_no = [0.974798979,0.703695 , 0.821747047, np.nan]
# cfl_no = [0.841760555,0.6391775 , 0.671624971,np.nan]
 
# std_zzz_no_p = [0.029257302, 0.024246862, 0.03886734, 0.034597559]
# std_zzz_no = [0.029257302, 0.024246862, 0.03886734, 0.034597559]
# std_fedavg_no = [0.015458903, 0.032424683, 0.021858135, 0.026939292]
# std_ifca_no = [0.060786177, 0.071028878, 0.029856031, 0.039467527]
# std_fedrc_no = [0.028691118, 0.029926574, 0.025152141, 0.019549612]
# std_fedem_no = [0.027751589, 0.026812692, 0.037942305, 0.029208152]
# std_fesem_no = [0.04587346, 0.032393998, 0.016258832, 0.01381074]
# std_feddrift_no = [0.001993407, 0.035090051, 0.032772943,np.nan]
# std_cfl_no = [0.016459322, 0.017586889, 0.020399327,np.nan]
# Data_load_no_inference
zzz_no_p = [0.977307256, 0.901964068, 0.903567816, 0.904978363]
zzz_no = [0.935082162, 0.816508762, 0.82592487, 0.854686703]
fedavg_no = [0.844397593, 0.700330634, 0.661952839, 0.658734032]
ifca_no = [0.809057795, 0.760109587, 0.762366395, 0.744451775]
fedrc_no = [0.747058886, 0.310470395, 0.283623699, 0.285470752]
fedem_no = [0.75658316, 0.326672057, 0.292381201, 0.299923074]
fesem_no = [0.826859755, 0.754151627, 0.72082539, 0.684889664]
feddrift_no = [0.974798979, 0.703695, 0.821747047, np.nan]
cfl_no = [0.841760555, 0.6391775, 0.671624971, np.nan]

std_zzz_no_p = [0.002245121, 0.004713994, 0.003961318, 0.00329857]
std_zzz_no = [0.020178264, 0.031774986, 0.035534008, 0.029971903]
std_fedavg_no = [0.015458903, 0.032424683, 0.021858135, 0.026939292]
std_ifca_no = [0.060786177, 0.071028878, 0.029856031, 0.039467527]
std_fedrc_no = [0.028691118, 0.029926574, 0.025152141, 0.019549612]
std_fedem_no = [0.027751589, 0.026812692, 0.037942305, 0.029208152]
std_fesem_no = [0.04587346, 0.032393998, 0.016258832, 0.01381074]
std_feddrift_no = [0.001993407, 0.035090051, 0.032772943, np.nan]
std_cfl_no = [0.016459322, 0.017586889, 0.020399327, np.nan]
 
 
# Data load real inference
# zzz_p = [0.956716217, 0.879598748, 0.847792465, 0.843035426]
# zzz = [0.956716217, 0.879598748, 0.847792465, 0.843035426]
# fedavg = [0.890870124, 0.753380846, 0.707505452, 0.704921209]
# ifca = [0.733307476, 0.709316806, 0.703150674, 0.693168921]
# fedrc = [0.758460647, 0.37597916, 0.340760224, 0.343029983]
# fedem = [0.752704081, 0.36899784, 0.32267421, 0.339273023]
# fesem = [0.768386273, 0.746337983, 0.7063584, 0.708714788]
# feddrift = [0.556212845, 0.629115056, 0.519060594, np.nan]
# cfl = [0.887074074, 0.648354222, 0.718713294, np.nan]
 
# std_zzz_p = [0.021848474, 0.031174757, 0.019063286, 0.023073598]
# std_zzz = [0.021848474, 0.031174757, 0.019063286, 0.023073598]
# std_fedavg = [0.009441559, 0.036420652, 0.024454846, 0.030378255]
# std_ifca = [0.042698375, 0.042151103, 0.030092447, 0.020952999]
# std_fedrc = [0.029222156, 0.033590148, 0.028917407, 0.021615494]
# std_fedem = [0.024121479, 0.026784699, 0.042449158, 0.02081338]
# std_fesem = [0.042339274, 0.018881085, 0.015327673, 0.014635534]
# std_feddrift = [0.029962109, 0.024597799, 0.025541083, np.nan]
# std_cfl = [0.011792477, 0.024528729, 0.022797711, np.nan]
# Data_load
zzz_p = [0.980596422, 0.904845855, 0.892441294, 0.890557302]
zzz = [0.953479785, 0.846240382, 0.847237617, 0.844495564]
fedavg = [0.890870124, 0.753380846, 0.707505452, 0.704921209]
ifca = [0.733307476, 0.709316806, 0.703150674, 0.693168921]
fedrc = [0.758460647, 0.37597916, 0.340760224, 0.343029983]
fedem = [0.752704081, 0.36899784, 0.32267421, 0.339273023]
fesem = [0.768386273, 0.746337983, 0.7063584, 0.708714788]
feddrift = [0.556212845, 0.629115056, 0.519060594, np.nan]
cfl = [0.887074074, 0.648354222, 0.718713294, np.nan]

std_zzz_p = [0.002471103, 0.01665148, 0.014192074, 0.0207911]
std_zzz = [0.015375, 0.021809282, 0.016003587, 0.021838005]
std_fedavg = [0.009441559, 0.036420652, 0.024454846, 0.030378255]
std_ifca = [0.042698375, 0.042151103, 0.030092447, 0.020952999]
std_fedrc = [0.029222156, 0.033590148, 0.028917407, 0.021615494]
std_fedem = [0.024121479, 0.026784699, 0.042449158, 0.02081338]
std_fesem = [0.042339274, 0.018881085, 0.015327673, 0.014635534]
std_feddrift = [0.029962109, 0.024597799, 0.025541083, np.nan]
std_cfl = [0.011792477, 0.024528729, 0.022797711, np.nan]

# TIME
# zzz_t_p = [4.486875235, 6.392059776, 8.084011098, 9.148540136]
# zzz_t = [4.486875235, 6.392059776, 8.084011098, 9.148540136]
# fedavg_t = [4.427270935, 6.327507735, 7.314243347, 8.582527778]
# ifca_t = [6.646702702, 8.573628254, 10.58251367, 10.80870279]
# fedrc_t = [7.825197314, 10.01341341, 12.07748252, 12.28466941]
# fedem_t = [7.747303473, 9.832126544, 11.97560201, 11.75715246]
# fesem_t = [6.582894659, 8.370850845, 9.982587225, 10.30899594]
# feddrift_t = [7.948367232, 13.98953398, 16.33708903, np.nan]
# cfl_t = [7.543225198, 14.02955253, 16.238312, np.nan]
# Time means and standard deviations
zzz_t_p = [53.52512121, 109.8875, 233.7025, 513.9125]
zzz_t = [42.80348602, 105.1875, 230.315, 516.7475]
fedavg_t = [21.515, 80.31, 159.15, 383.3525]
ifca_t = [100.1975, 380.995, 1533.395, 1793.675]
fedrc_t = [226.7875, 1033.565, 4321.9975, 4989.465]
fedem_t = [214.8675, 911.5175, 4027.313344, 3461.43]
fesem_t = [95.8625, 331.0375, 1011.715, 1268.5775]
feddrift_t = [247, 16265.5725, 82785.41664, np.nan]
cfl_t = [186.525, 16723.075, 77307.03053, np.nan]

std_zzz_t_p = [3.961963244, 1.564696456, 20.32033403, 17.43486951]
std_zzz_t = [4.296827318, 5.673486582, 55.6137928, 60.64172697]
std_fedavg_t = [0.992899794, 5.105097942, 18.54524198, 9.158983295]
std_ifca_t = [6.281070769, 29.80758544, 107.1570777, 210.4928746]
std_fedrc_t = [25.3361846, 52.36254339, 883.7921492, 1813.319851]
std_fedem_t = [21.26665759, 94.64437516, 1528.619042, 190.6309528]
std_fesem_t = [8.128303944, 59.67706406, 89.8873164, 189.2480988]
std_feddrift_t = [27.26059427, 3413.589585, 12594.12174, np.nan]
std_cfl_t = [23.82771496, 4262.829965, 8521.717811, np.nan]


# Transform time means into powers of 2
zzz_t_p_log2 = np.log2(zzz_t_p)
zzz_t_log2 = np.log2(zzz_t)
fedavg_t_log2 = np.log2(fedavg_t)
ifca_t_log2 = np.log2(ifca_t)
fedrc_t_log2 = np.log2(fedrc_t)
fedem_t_log2 = np.log2(fedem_t)
fesem_t_log2 = np.log2(fesem_t)
feddrift_t_log2 = [np.log2(val) if val is not np.nan else np.nan for val in feddrift_t]
cfl_t_log2 = [np.log2(val) if val is not np.nan else np.nan for val in cfl_t]

# Transform time standard deviations into powers of 2
std_zzz_t_p_log2 = [np.log2(val) if np.log2(val) > 0 else 0 for val in std_zzz_t_p]
std_zzz_t_log2 = [np.log2(val) if np.log2(val) > 0 else 0 for val in std_zzz_t]
std_fedavg_t_log2 = [np.log2(val) if np.log2(val) > 0 else 0 for val in std_fedavg_t]
std_ifca_t_log2 = [np.log2(val) if np.log2(val) > 0 else 0 for val in std_ifca_t]
std_fedrc_t_log2 = [np.log2(val) if np.log2(val) > 0 else 0 for val in std_fedrc_t]
std_fedem_t_log2 = [np.log2(val) if np.log2(val) > 0 else 0 for val in std_fedem_t]
std_fesem_t_log2 = [np.log2(val) if np.log2(val) > 0 else 0 for val in std_fesem_t]
std_feddrift_t_log2 = [np.log2(val) if val is not np.nan and np.log2(val) > 0 else 0 for val in std_feddrift_t]
std_cfl_t_log2 = [np.log2(val) if val is not np.nan and np.log2(val) > 0 else 0 for val in std_cfl_t]


 
 
# change here
two_column = True
setup_icml_plot(two_column=two_column)
 
 
# Create a figure with three horizontal subplots
fig, axes = plt.subplots(1, 3, figsize=(7, 2.35))  # Maintain two-column format with horizontal subplots
 
# Plot known cluster
axes[0].errorbar(x, zzz_no_p, yerr=std_zzz_no_p, label="_nolegend_", color="black", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="black")
axes[0].errorbar(x, zzz_no, yerr=std_zzz_no, label="_nolegend_", color="blue", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="blue")
axes[0].errorbar(x, fedavg_no, yerr=std_fedavg_no, label="_nolegend_", color="orange", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="orange")
axes[0].errorbar(x, ifca_no, yerr=std_ifca_no, label="_nolegend_", color="green", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="green")
axes[0].errorbar(x, fedrc_no, yerr=std_fedrc_no, label="_nolegend_", color="red", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="red")
axes[0].errorbar(x, fedem_no, yerr=std_fedem_no, label="_nolegend_", color="purple", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="purple")
axes[0].errorbar(x, fesem_no, yerr=std_fesem_no, label="_nolegend_", color="brown", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="brown")
axes[0].errorbar(x, feddrift_no, yerr=std_feddrift_no, label="_nolegend_", color="pink", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="pink")
axes[0].errorbar(x, cfl_no, yerr=std_cfl_no, label="_nolegend_", color="cyan", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="cyan")
axes[0].plot(x, fedavg_no, label="FedAvg", color="orange", marker='o', markersize=2)
axes[0].plot(x, ifca_no, label="IFCA", color="green", marker='o', markersize=2)
axes[0].plot(x, fedrc_no, label="FedRC", color="red", marker='o', markersize=2)
axes[0].plot(x, fedem_no, label="FedEM", color="purple", marker='o', markersize=2)
axes[0].plot(x, fesem_no, label="FeSEM", color="brown", marker='o', markersize=2)
axes[0].plot(x, feddrift_no, label="FedDrift", color="pink", marker='o', markersize=2)
axes[0].plot(x, cfl_no, label="CFL", color="cyan", marker='o', markersize=2)
axes[0].plot(x, zzz_no, label="FLUX", color="blue", marker='o', markersize=2)
axes[0].plot(x, zzz_no_p, label="FLUX-prior", color="black", marker='o', markersize=2)
axes[0].set_title("Known Association")
axes[0].set_xlabel("Number of clients")
axes[0].set_ylabel("Accuracy")
axes[0].set_xticks(x)
axes[0].set_xticklabels([5, 25, 50, 100])
axes[0].set_ylim(0.2, 1)
axes[0].legend(loc="lower left")
 
# Plot testing phase inference
axes[1].errorbar(x, zzz_p, yerr=std_zzz_p, label="_nolegend_", color="black", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="black")
axes[1].errorbar(x, zzz, yerr=std_zzz, label="_nolegend_", color="blue", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="blue")
axes[1].errorbar(x, fedavg, yerr=std_fedavg, label="_nolegend_", color="orange", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="orange")
axes[1].errorbar(x, ifca, yerr=std_ifca, label="_nolegend_", color="green", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="green")
axes[1].errorbar(x, fedrc, yerr=std_fedrc, label="_nolegend_", color="red", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="red")
axes[1].errorbar(x, fedem, yerr=std_fedem, label="_nolegend_", color="purple", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="purple")
axes[1].errorbar(x, fesem, yerr=std_fesem, label="_nolegend_", color="brown", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="brown")
axes[1].errorbar(x, feddrift, yerr=std_feddrift, label="_nolegend_", color="pink", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="pink")
axes[1].errorbar(x, cfl, yerr=std_cfl, label="_nolegend_", color="cyan", marker="o", markersize=2,
                 elinewidth=0.8, capsize=2, alpha=0.2, ecolor="cyan")
axes[1].plot(x, fedavg, label="FedAvg", color="orange", marker='o', markersize=2)
axes[1].plot(x, ifca, label="IFCA", color="green", marker='o', markersize=2)
axes[1].plot(x, fedrc, label="FedRC", color="red", marker='o', markersize=2)
axes[1].plot(x, fedem, label="FedEM", color="purple", marker='o', markersize=2)
axes[1].plot(x, fesem, label="FeSEM", color="brown", marker='o', markersize=2)
axes[1].plot(x, feddrift, label="FedDrift", color="pink", marker='o', markersize=2)
axes[1].plot(x, cfl, label="CFL", color="cyan", marker='o', markersize=2)
axes[1].plot(x, zzz, label="FLUX", color="blue", marker='o', markersize=2)
axes[1].plot(x, zzz_p, label="FLUX-prior", color="black", marker='o', markersize=2)
axes[1].set_title("Test Phase")
axes[1].set_xlabel("Number of clients")
axes[1].set_ylabel("Accuracy")
axes[1].set_xticks(x)
axes[1].set_xticklabels([5, 25, 50, 100])
axes[1].set_ylim(0.2, 1)
axes[1].legend(loc="lower left")
 
 
# Plot Time
axes[2].plot(x, fedavg_t_log2, label="FedAvg", color="orange", marker='o', markersize=2)
axes[2].plot(x, ifca_t_log2, label="IFCA", color="green", marker='o', markersize=2)
axes[2].plot(x, fedrc_t_log2, label="FedRC", color="red", marker='o', markersize=2)
axes[2].plot(x, fedem_t_log2, label="FedEM", color="purple", marker='o', markersize=2)
axes[2].plot(x, fesem_t_log2, label="FeSEM", color="brown", marker='o', markersize=2)
axes[2].plot(x, feddrift_t_log2, label="FedDrift", color="pink", marker='o', markersize=2)
axes[2].plot(x, cfl_t_log2, label="CFL", color="cyan", marker='o', markersize=2)
axes[2].plot(x, zzz_t_log2, label="FLUX", color="blue", marker='o', markersize=2)
axes[2].plot(x, zzz_t_p_log2, label="FLUX-prior", color="black", marker='o', markersize=2)
axes[2].set_title("Training Time")
axes[2].set_xlabel("Number of clients")
axes[2].set_ylabel("Time (seconds)")
axes[2].set_xticks(x)
axes[2].set_xticklabels([5, 25, 50, 100])
axes[2].set_ylim(3, 19)
y_ticks = [4, 6,8,10,12,14 ,16,18]  # Actual powers of 2 in your data
axes[2].set_yticks(y_ticks)
axes[2].set_yticklabels([r"$2^{4}$", r"$2^{6}$", r"$2^{8}$", r"$2^{10}$", r"$2^{12}$", r"$2^{14}$", r"$2^{16}$", r"$2^{18}$"])  # LaTeX power notation
 
axes[2].legend(loc="upper left")

# Plot Time (log2-transformed with error bars)
# axes[2].errorbar(x, fedavg_t_log2, yerr=std_fedavg_t_log2, label="_nolegend_", color="orange", marker="o", markersize=2,
#                  elinewidth=0.8, capsize=2, alpha=0.2, ecolor="orange")
# axes[2].errorbar(x, ifca_t_log2, yerr=std_ifca_t_log2, label="_nolegend_", color="green", marker="o", markersize=2,
#                  elinewidth=0.8, capsize=2, alpha=0.2, ecolor="green")
# axes[2].errorbar(x, fedrc_t_log2, yerr=std_fedrc_t_log2, label="_nolegend_", color="red", marker="o", markersize=2,
#                  elinewidth=0.8, capsize=2, alpha=0.2, ecolor="red")
# axes[2].errorbar(x, fedem_t_log2, yerr=std_fedem_t_log2, label="_nolegend_", color="purple", marker="o", markersize=2,
#                  elinewidth=0.8, capsize=2, alpha=0.2, ecolor="purple")
# axes[2].errorbar(x, fesem_t_log2, yerr=std_fesem_t_log2, label="_nolegend_", color="brown", marker="o", markersize=2,
#                  elinewidth=0.8, capsize=2, alpha=0.2, ecolor="brown")
# axes[2].errorbar(x, feddrift_t_log2, yerr=std_feddrift_t_log2, label="_nolegend_", color="pink", marker="o", markersize=2,
#                  elinewidth=0.8, capsize=2, alpha=0.2, ecolor="pink")
# axes[2].errorbar(x, cfl_t_log2, yerr=std_cfl_t_log2, label="_nolegend_", color="cyan", marker="o", markersize=2,
#                  elinewidth=0.8, capsize=2, alpha=0.2, ecolor="cyan")
# axes[2].errorbar(x, zzz_t_log2, yerr=std_zzz_t_log2, label="_nolegend_", color="blue", marker="o", markersize=2,
#                  elinewidth=0.8, capsize=2, alpha=0.2, ecolor="blue")
# axes[2].errorbar(x, zzz_t_p_log2, yerr=std_zzz_t_p_log2, label="_nolegend_", color="black", marker="o", markersize=2,
#                  elinewidth=0.8, capsize=2, alpha=0.2, ecolor="black")
# axes[2].plot(x, fedavg_t_log2, label="FedAvg", color="orange", marker='o', markersize=2)
# axes[2].plot(x, ifca_t_log2, label="IFCA", color="green", marker='o', markersize=2)
# axes[2].plot(x, fedrc_t_log2, label="FedRC", color="red", marker='o', markersize=2)
# axes[2].plot(x, fedem_t_log2, label="FedEM", color="purple", marker='o', markersize=2)
# axes[2].plot(x, fesem_t_log2, label="FeSEM", color="brown", marker='o', markersize=2)
# axes[2].plot(x, feddrift_t_log2, label="FedDrift", color="pink", marker='o', markersize=2)
# axes[2].plot(x, cfl_t_log2, label="CFL", color="cyan", marker='o', markersize=2)
# axes[2].plot(x, zzz_t_log2, label="Ours", color="blue", marker='o', markersize=2)
# axes[2].plot(x, zzz_t_p_log2, label="Ours (prior)", color="black", marker='o', markersize=2)
# axes[2].set_title("Training Time (5 Folds)")
# axes[2].set_xlabel("Number of clients")
# axes[2].set_ylabel("Time (log2 scale - seconds)")
# axes[2].set_xticks(x)
# axes[2].set_xticklabels([5, 25, 50, 100])
# axes[2].set_ylim(3, 19)
# y_ticks = [4, 6, 8, 10, 12, 14, 16, 18]  # Powers of 2
# axes[2].set_yticks(y_ticks)
# axes[2].set_yticklabels([r"$2^{4}$", r"$2^{6}$", r"$2^{8}$", r"$2^{10}$", r"$2^{12}$", r"$2^{14}$", r"$2^{16}$", r"$2^{18}$"])
# axes[2].legend(loc="upper right")
 
# Adjust layout
plt.tight_layout()
plt.savefig("N_clients.pdf", bbox_inches="tight")
plt.show()
