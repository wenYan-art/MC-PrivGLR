import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from typing import List
import public.config as cfg

# Create folders
def create_folders():
    os.makedirs(f"results/{cfg.default_path}", exist_ok=True)
    os.makedirs(f"histories/{cfg.default_path}", exist_ok=True)
    os.makedirs(f"checkpoints/{cfg.default_path}", exist_ok=True)
    os.makedirs(f"images/{cfg.default_path}", exist_ok=True)

    return cfg.default_path

# define device
def check_gpu(client_id:int = 0):
    torch.manual_seed(cfg.random_seed)
    if cfg.gpu == -1:
        device = 'cpu'
    elif torch.cuda.is_available():
        if cfg.gpu == -2: # multiple gpu
            # assert client_id >=0, "client_id must be passed to select the respective GPU"
            n_total_gpus = torch.cuda.device_count()
            if client_id < 15:
                device = "cuda:2"
            else:
                device = "cuda:1"
            # device = 'cuda:' + str(int(client_id % n_total_gpus))
        else:
            device = 'cuda:' + str(cfg.gpu)
        torch.cuda.manual_seed_all(cfg.random_seed) 
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.mps.manual_seed(cfg.random_seed)
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    return device

# plot and save plot on server side
def plot_loss_and_accuracy(
        loss: List[float],
        accuracy: List[float],
        show: bool = True,
        fold=0):
    
    # # Plot loss separately
    plt.figure(figsize=(12, 6))
    plt.plot(loss, label='Loss', color='blue')
    min_loss_index = loss.index(min(loss))
    plt.scatter(min_loss_index, loss[min_loss_index], color='red', marker='*', s=100, label='Min Loss')
    
    # Labels and title for loss
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Distributed Loss (Weighted Average on Test-Set)')
    plt.legend()
    
    # Save the loss plot
    loss_plot_path = f"images/{cfg.default_path}/{cfg.non_iid_type}_loss_n_clients_{cfg.n_clients}_n_rounds_{cfg.n_rounds}_fold_{fold}.png"
    plt.savefig(loss_plot_path)
    if show:
        plt.show()
    else:
        plt.close()

    # Plot accuracy separately
    plt.figure(figsize=(12, 6))
    plt.plot(accuracy, label='Accuracy', color='green')
    max_accuracy_index = accuracy.index(max(accuracy))
    plt.scatter(max_accuracy_index, accuracy[max_accuracy_index], color='orange', marker='*', s=100, label='Max Accuracy')
    
    # Labels and title for accuracy
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Distributed Accuracy (Weighted Average on Test-Set)')
    plt.legend()
    
    # Save the accuracy plot
    accuracy_plot_path = f"images/{cfg.default_path}/{cfg.non_iid_type}_accuracy_n_clients_{cfg.n_clients}_n_rounds_{cfg.n_rounds}_fold_{fold}.png"
    plt.savefig(accuracy_plot_path)
    if show:
        plt.show()
    else:
        plt.close()

    # Print out server-side information
    print(f"\n\033[1;34mServer Side\033[0m \nMinimum Loss occurred at round {min_loss_index + 1} with a loss value of {loss[min_loss_index]:.3f} \nMaximum Accuracy occurred at round {max_accuracy_index + 1} with an accuracy value of {accuracy[max_accuracy_index]*100:.2f}\n")
    
    return min_loss_index + 1, max_accuracy_index + 1

# Cluster plot
def cluster_plot(X_reduced, cluster_labels, client_cid, server_round, name="KMeans"):
    # Create a folder to save the plots
    if not os.path.exists(f"images/{cfg.default_path}/plots_descriptors"):
        os.makedirs(f"images/{cfg.default_path}/plots_descriptors")
    
    # number of clusters - only number of cluster_labels - no string element
    # n_clusters = np.unique([n for n in cluster_labels if n.isnumeric()]).shape[0]
    n_clusters = np.unique(cluster_labels).shape[0]
    
    # Plot the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=cluster_labels, palette="deep", legend="full", s=100)
    plt.title(f'{name} ({n_clusters} Clusters) - R.{server_round}', fontsize=18)
    plt.xlabel('PC1', fontsize=16)
    plt.ylabel('PC2', fontsize=16)
    # Annotate client id
    for i, cid in enumerate(client_cid):
        plt.text(X_reduced[i, 0], X_reduced[i, 1], str(cid), fontsize=10, ha='right')
    # Save the plot
    plt.savefig(f"images/{cfg.default_path}/plots_descriptors/{name.lower()}_cluster_visualization_{server_round}.png")
    plt.close()
    
# Plot the elbow and silhouette scores
def plot_elbow_and_silhouette(range_n_clusters, inertia, silhouette_scores, server_round):
    # Create a folder to save the plots
    if not os.path.exists(f"images/{cfg.default_path}/plots_descriptors"):
        os.makedirs(f"images/{cfg.default_path}/plots_descriptors")
        
    # Create figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))  # Two plots side by side, width is larger (20) to accommodate both plots

    # Plot inertia (Elbow Method) on the first subplot
    axs[0].plot(range_n_clusters, inertia, marker='o', label='Inertia')
    axs[0].set_title(f'Elbow Method (Optimal Cluster: {range_n_clusters[np.argmin(inertia)]}) - R.{server_round}', fontsize=18)
    axs[0].set_xlabel('Number of clusters', fontsize=16)
    axs[0].set_ylabel('Inertia', fontsize=16)

    # Plot silhouette scores on the second subplot
    axs[1].plot(range_n_clusters, silhouette_scores, marker='o', label='Silhouette Score')
    axs[1].set_title(f'Silhouette Scores (Optimal Cluster: {range_n_clusters[np.argmax(silhouette_scores)]}) - R.{server_round}', fontsize=18)
    axs[1].set_xlabel('Number of clusters', fontsize=16)
    axs[1].set_ylabel('Silhouette Score', fontsize=16)

    # Save the combined figure to the appropriate directory
    plt.savefig(f"images/{cfg.default_path}/plots_descriptors/elbow_and_silhouette_{server_round}.png")
    plt.close()

# Get cur dataset in_channels
def get_in_channels():
    if cfg.dataset_name == "CheXpert":
        return 1
    else:
        for file_name in ['../data/cur_datasets/client_1.npy', '../data/cur_datasets/client_1_round_-1.npy']:
            if os.path.exists(file_name):
                cur_data = np.load(file_name, allow_pickle=True).item()
                break
        cur_features = cur_data['train_features'] if not cfg.training_drifting else cur_data['features']

        return 3 if len(cur_features.shape) == 4 else 1

def set_seed(seed):
    # Set seed for torch
    torch.manual_seed(seed)
    
    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(cfg.random_seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set deterministic behavior for CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set PYTHONHASHSEED
    os.environ['PYTHONHASHSEED'] = str(seed)

# Calculate centroids
def calculate_centroids(data: np.ndarray,
                        clustering_method,
                        cluster_labels,
                        save=True):
    """
    Calculate the centroids of the clusters.
    
    Args:
    data: np.ndarray
        The data points.
    clustering_method: sklearn.cluster object
        The clustering method used.
    cluster_labels: np.ndarray
        The cluster labels.
    
    Returns:
        A dictionary containing the cluster label as the key and the centroid as the value.
    """
    
    # Kmeans
    if cfg.cfl_oneshot_CLIENT_CLUSTER_METHOD == 1 or cfg.cfl_oneshot_CLIENT_CLUSTER_METHOD == 5:
        centroids = clustering_method.cluster_centers_
        centroids_dict = {label: np.array(centroid) for label, centroid in zip(np.unique(cluster_labels), centroids)}
    
    # DBSCAN and HDBSCAN, DBSCAN_no_outliers
    elif cfg.cfl_oneshot_CLIENT_CLUSTER_METHOD in [2, 3, 4]:
        centroids_dict = {}
        for label in np.unique(cluster_labels):
            cluster_points = data[cluster_labels == label]
            centroids_dict[label] = np.array(cluster_points.mean(axis=0))
    
    # Save
    # print(f"Centroids: {centroids_dict}")
    if save:
        path = f"results/{cfg.default_path}"
        np.save(f"{path}/centroids_{cfg.non_iid_type}_n_clients_{cfg.n_clients}.npy", centroids_dict, allow_pickle=True)
    
    return centroids_dict

def plot_all_clients_metrics(n_clients=cfg.n_clients, save=True, show=False, fold=""):
    # Loss
    plt.figure(figsize=(12, 6))
    for client_id in range(n_clients):
        # Load metrics for each client
        metrics_path = f"results/{cfg.default_path}/client_{client_id}_metrics.npy"
        metrics = np.load(metrics_path, allow_pickle=True).item()
        plt.plot(metrics["rounds"], metrics["loss"], label=f'Client {client_id} Loss')
    
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Loss per Round for All Clients')
    plt.legend()

    print(f"Saving loss plot for all clients OUTSIDE...")

    if save:
        print(f"Saving loss plot for all clients...")
        print(f"images/{cfg.default_path}/all_clients_loss_fold_{fold}.png")
        plt.savefig(f"images/{cfg.default_path}/all_clients_loss_fold_{fold}.png")
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(12, 6))

    for client_id in range(n_clients):
        # Load metrics for each client
        metrics_path = f"results/{cfg.default_path}/client_{client_id}_metrics.npy"
        metrics = np.load(metrics_path, allow_pickle=True).item()
        plt.plot(metrics["rounds"], metrics["accuracy"], label=f'Client {client_id} Accuracy')
    
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Round for All Clients')
    plt.legend()

    if save:
        plt.savefig(f"images/{cfg.default_path}/all_clients_accuracy_fold_{fold}.png")
    if show:
        plt.show()
    else:
        plt.close()
        
