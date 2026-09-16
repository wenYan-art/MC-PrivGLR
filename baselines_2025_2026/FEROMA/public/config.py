# Overall settings
k_folds = 2 # number of folds for cross-validation, if 1, no cross-validation
strategy = 'feroma' # ['fedavg', 'feroma']
random_seed = 42
gpu = 0 # set the GPU to use, if -1 use CPU, -2 for multigpus
n_clients = 10
n_samples_clients = -1 # if -1, use all samples

# differential privacy on the descriptors
differential_privacy_descriptors = False
epsilon = 10.0
# sensitivity = 1.0 # automatically calculated

# FEROMA settings
distance_function = "euclidean" # ['cosine', 'euclidean']
n_stochastic_sampling = 3 # number of times to sample the data for the stochastic sampling descriptor extraction; 0 for no sampling
extended_descriptors = True #mean and std 
weighted_metric_descriptors = False
selected_descriptors = "Px_label_long" # Options: "Px", "Py", "Pxy", "Px_cond", "Pxy_cond", "Px_label_long", "Px_label_short" for training time
cfl_oneshot_CLIENT_SCALING_METHOD = 1 # ['Ours', 'weighted', 'none']
n_test_sample_per_class = 20
distance_visualization = True

# Dynamic dataset (ANDA) settings
drifting_type = 'trDR_teND'
dataset_name = "MNIST" # ["CIFAR10", "CIFAR100", "MNIST", "FMNIST", "CheXpert"]
verbose = True
count_labels = True
plot_clients = False
non_iid_type = 'Px'         # ['Px','Py','Px_y','Py_x'] 

# Training model settings
model_name = "ResNet9"   # ["LeNet5", "ResNet9"]
batch_size = 64
test_batch_size = 64
client_eval_ratio = 0.2
n_rounds = 10
local_epochs = 2
lr = 0.005
momentum = 0.9

# self-defined settings
n_classes_dict = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "MNIST": 10,
    "FMNIST": 10,
    "CheXpert": 14,
}
n_classes = n_classes_dict[dataset_name]

input_size_dict = {
    "CIFAR10": (32, 32),
    "CIFAR100": (32, 32),
    "MNIST": (28, 28),
    "FMNIST": (28, 28),
    "CheXpert": (64, 64),
}
input_size = input_size_dict[dataset_name]

acceptable_accuracy = {
    "CIFAR10": 0.5,
    "CIFAR100": 0.1,
    "MNIST": 0.8,
    "FMNIST": 0.8,
    "CheXpert": 0.7,
}
th_accuracy = acceptable_accuracy[dataset_name]
training_drifting = False if drifting_type in ['static', 'trND_teDR'] else True # to be identified
default_path = f"{random_seed}/{model_name}/{dataset_name}/{drifting_type}"

# FL settings - Communications
port = '8018'
ip = '0.0.0.0' # Local Host=0.0.0.0, or IP address of the server

# Advance One-shot settings
len_metric_descriptor =  n_classes
n_metrics_descriptors = 2 if extended_descriptors else 1
len_latent_space_descriptor = 1 * len_metric_descriptor   # modify this to change the latent space size
n_latent_space_descriptors = 2 if extended_descriptors else 1
