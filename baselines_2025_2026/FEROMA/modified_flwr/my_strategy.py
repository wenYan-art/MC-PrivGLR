# Libraries
import flwr as fl
from modified_flwr import app
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import Parameters, Scalar, Metrics
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes
import argparse
import torch
from torch.utils.data import DataLoader
import os
from logging import WARNING
from flwr.common.logger import log
from collections import OrderedDict
import json
import time
import pandas as pd
import config as cfg
import utils
import models
from sklearn.model_selection import train_test_split
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitRes,
    FitIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common import NDArray, NDArrays
from functools import reduce
from flwr.server.client_manager import ClientManager
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from modified_flwr.server import Server


# Define the max latent space as global variable
max_latent_space = 2


# Config_client
def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "local_epochs": cfg.local_epochs,
        "tot_rounds": cfg.n_rounds,
        "min_latent_space": 0,
        "max_latent_space": max_latent_space,
    }
    return config

# Custom weighted average function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    # validities = [num_examples * m["validity"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples

def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

def aggregate_fit(
    self,
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    """Aggregate fit results using weighted average."""
    if not results:
        return None, {}
    # Do not aggregate if there are failures and failures are not accepted
    if not self.accept_failures and failures:
        return None, {}

    # Convert results
    weights_results = [
        (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        for _, fit_res in results
    ]
    parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

    # Aggregate custom metrics if aggregation fn was provided
    metrics_aggregated = {}
    if self.fit_metrics_aggregation_fn:
        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
    elif server_round == 1:  # Only log this warning once
        log(WARNING, "No fit_metrics_aggregation_fn provided")

    return parameters_aggregated, metrics_aggregated

# Custom strategy to save model after each round
class MyStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.dataset = dataset
        self.client_cid_list = []
        self.aggregated_cluster_parameters = []
        self.cluster_labels = {}
        self.aggregated_parameters_global = None
        self.cluster_done = False
        self.cluster_do = False

    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        
        
        ################################################################################
        # Client descriptors analysis
        ################################################################################
        # Update client cid list   
        self.client_cid_list = []
        for client in results:
            self.client_cid_list.append(client[0].cid)     # Automatically assigned cid by Flower
        
        # Extract client descriptors
        client_descr = []
        client_cid = []
        for _, res in results:
            # res.num_examples, res.metrics
            loss_pc = json.loads(res.metrics["loss_pc"])
            latent_space = json.loads(res.metrics["latent_space"])
            client_cid.append(res.metrics["cid"])
            # concatenate the descriptors
            client_descr.append(loss_pc + latent_space)
        
        # Normalizing descriptor matrix
        client_descr = np.array(client_descr)
        # 1. Normalize each column between 0 and 1 #TODO: test both 1 and 2 methods
        scaler = MinMaxScaler() # StandardScaler()
        client_descr_scaled_1 = scaler.fit_transform(client_descr)
        # print(f"scaled_data: {client_descr_scaled_1}")
        # 2. Normalize by group of descriptors #TODO: test both 1 and 2 methods
        loss_pc = client_descr[:, :cfg.n_classes]
        latent_space = client_descr[:, cfg.n_classes:]
        scaled_loss_pc = scaler.fit_transform(loss_pc.reshape(-1, 1)).reshape(loss_pc.shape)  
        latent_space_pc = scaler.fit_transform(latent_space.reshape(-1, 1)).reshape(latent_space.shape)
        client_descr_scaled_2 = np.hstack((scaled_loss_pc, latent_space_pc)) # TODO: we can also weight them (loss and latent) differently here, by multiplying them by a factor
        # print(f"scaled_data_2: {client_descr_scaled_2}")
        
        # TODO: choose the best method for normalization
        client_descr_scaled = client_descr_scaled_2
        
        
        ###############################################################################            
        # Clustering
        ###############################################################################
        if self.cluster_do:
            print(f"\033[91mRound {server_round} - Clustering clients...\033[0m")
            self.cluster_done = True
            self.cluster_do = False
            
            # Range of cluster numbers to try
            range_n_clusters = range(2, client_descr_scaled.shape[0])  # Adjust based on your data size

            # Store inertia (sum of squared distances to centroids) and silhouette scores
            inertia = []
            silhouette_scores = []
            for n_clusters in range_n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(client_descr_scaled)
                inertia.append(kmeans.inertia_)
                
                # Calculate silhouette score and append
                silhouette_avg = silhouette_score(client_descr_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
        
            # # Plot inertia (Elbow Method) and silhouette scores
            utils.plot_elbow_and_silhouette(range_n_clusters, inertia, silhouette_scores, server_round)

            # Identify the best number of clusters based on the highest silhouette score
            best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]

            # Apply PCA to reduce the data to 2D for visualization
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(client_descr_scaled)

            # Apply KMeans with the best number of clusters
            kmeans_best = KMeans(n_clusters=best_n_clusters, random_state=42)
            cluster_labels_kmeans = kmeans_best.fit_predict(client_descr_scaled)
            # print(f"KMeans cluster_labels: {cluster_labels_kmeans}")
            # Plot the identified clusters with the best score/number of clusters
            utils.cluster_plot(X_reduced, cluster_labels_kmeans, client_cid, server_round, name="KMeans")

            # Apply DBSCAN (doesn't require specifying the number of clusters)
            # clustering = DBSCAN(eps=0.5, min_samples=2)  # You can tune the parameters `eps` and `min_samples`
            # cluster_labels_dbscan = clustering.fit_predict(client_descr_scaled)
            # print(f"DBSCAN cluster_labels: {cluster_labels_dbscan}")
            # Plot the identified DBSCAN clusters
            # utils.cluster_plot(X_reduced, cluster_labels_dbscan, client_cid, server_round, name="DBSCAN")
            
            # HDBSCAN
            clustering = HDBSCAN(min_cluster_size=2)  # You can tune the parameters `min_cluster_size` and `min_samples`
            cluster_labels_hdbscan = clustering.fit_predict(client_descr_scaled) # Note negative values are outliers, here I make them positive for visualization
            if min(cluster_labels_hdbscan) < 0:
                cluster_labels_hdbscan = cluster_labels_hdbscan + abs(min(cluster_labels_hdbscan))
            # print(f"HDBSCAN cluster_labels after: {cluster_labels_hdbscan}")
            # Plot the identified HDBSCAN clusters
            utils.cluster_plot(X_reduced, cluster_labels_hdbscan, client_cid, server_round, name="HDBSCAN")
            
            # Choose the best clustering methods
            self.n_clusters = max(cluster_labels_hdbscan) + 1  # Best number of clusters
            self.cluster_labels = {cid: cluster_labels_hdbscan[i] for i, cid in enumerate(self.client_cid_list)}  # Best clustering method
            print(f"\033[91mRound {server_round} - Identified {self.n_clusters} clusters ({self.cluster_labels.values()})\033[0m")

        
        
        ################################################################################
        # Federated averaging aggregation
        ################################################################################
        # Federated averaging - from traditional code
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # i can skip this part when cluster_done is False, but i still need aggregated_parameters_global to pass
        aggregated_parameters_global = ndarrays_to_parameters(aggregate(weights_results))   # Global aggregation - traditional - no clustering
        self.aggregated_parameters_global = aggregated_parameters_global
        
        if self.cluster_done:
            # Split aggregation into clusters
            client_clusters = {i: [] for i in range(self.n_clusters)}
            for i, cluster in enumerate(self.cluster_labels.values()):
                client_clusters[cluster].append(weights_results[i])
                
            # Aggregate each cluster
            self.aggregated_cluster_parameters = []
            for cluster in client_clusters.values():
                    self.aggregated_cluster_parameters.append(ndarrays_to_parameters(aggregate(cluster)))
        
        # Aggregate custom metrics if aggregation fn was provided   NO FIT METRICS AGGREGATION FN PROVIDED - SKIPPED FOR NOW
        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
            
            
        ################################################################################
        # Save model
        ################################################################################
        if aggregated_parameters_global is not None:

            print(f"Saving round {server_round} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters_global)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            # Save the model
            torch.save(self.model.state_dict(), f"checkpoints/{cfg.model_name}/{cfg.dataset_name}/model_{server_round}.pth")
        
        if self.cluster_done:
            # Save the aggregated cluster models
            for i, aggregated_cluster_parameters in enumerate(self.aggregated_cluster_parameters):
                print(f"Saving round {server_round} aggregated_cluster_parameters_{i}...")
                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_cluster_parameters)
                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)
                # Save the model
                torch.save(self.model.state_dict(), f"checkpoints/{cfg.model_name}/{cfg.dataset_name}/model_{server_round}_cluster_{i}.pth")
        
        return aggregated_parameters_global, aggregated_metrics
    
    
    ############################################################################################################
    # Aggregate evaluation results
    ############################################################################################################
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        
        print(f"\033[93mRound {server_round} - Aggregated loss: {loss_aggregated} - Aggregated metrics: {metrics_aggregated}\033[0m")
        if metrics_aggregated["accuracy"] > cfg.th_accuracy and not self.cluster_done:
            self.cluster_do = True
            print(f"\033[93mRound {server_round} - Clustering flag: {self.cluster_do}\033[0m")
            
            # Saving evaluation model
            print(f"Saving round {server_round} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(self.aggregated_parameters_global)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            # Save the model
            torch.save(self.model.state_dict(), f"checkpoints/{cfg.model_name}/{cfg.dataset_name}/model_evaluation_clusters.pth")
        
        # Update the max_latent_space for the next round
        max_client_latent_space = max([res.metrics["max_latent_space"] for _, res in results])
        global max_latent_space 
        max_latent_space = 1.02 * max_client_latent_space 

        return loss_aggregated, metrics_aggregated
    

    ############################################################################################################
    # Configure fit - Send custom configuration to clients for training
    ############################################################################################################
    # Override configure_fit method to add custom configuration
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)      # Config sent to clients during training 
            
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # if cluster is not done use global otherwise use cluster aggregation
        if not self.cluster_done:
            # GLOBAL TRAINING
            fit_ins = FitIns(parameters, config)
            return [(client, fit_ins) for client in clients]
        else:           
            # CLUSTERED AGGREGATION - Instrucions for clients - custom fit_ins for each client 
            client_cids_sampled = [client.cid for client in clients]
            fit_ins = []
            for c in client_cids_sampled:
                fit_ins.append(FitIns(self.aggregated_cluster_parameters[self.cluster_labels[c]], config))
            return [(client, fit_ins[i]) for i, client in enumerate(clients)]
        
      
    
    ############################################################################################################
    # Configure evaluate - Send custom configuration to clients for evaluation
    ############################################################################################################  
    # Override configure_evaluate method to add custom configuration
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)      # Config sent to clients during evaluation
            
        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        if not self.cluster_done:
            # GLOBAL EVALUATION
            evaluate_ins = EvaluateIns(parameters, config)
            return [(client, evaluate_ins) for client in clients]
        else:
            # CLUSTERED AGGREGATION - SEND BACK FOR EVALUATION EACH CLUSTER MODEL TO THE RESPECTIVE CLIENT 
            client_cids_sampled = [client.cid for client in clients]
            evaluate_ins = []
            for c in client_cids_sampled:
                evaluate_ins.append(EvaluateIns(self.aggregated_cluster_parameters[self.cluster_labels[c]], config))
            return [(client, evaluate_ins[i]) for i, client in enumerate(clients)]
