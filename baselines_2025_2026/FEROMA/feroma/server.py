"""
This code implements the FedAvg, when it starts, the server waits for the clients to connect. When the established number 
of clients is reached, the learning process starts. The server sends the model to the clients, and the clients train the 
model locally. After training, the clients send the updated model back to the server. Then client models are aggregated 
with FedAvg. The aggregated model is then sent to the clients for the next round of training. The server saves the model 
and metrics after each round.

This is code is set to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
run the appopriate client code (client.py).

METHOD: in the first rounds, FedAvg is used until the global model reaches a pre-defined accuracy. After that the 
current global model is utilized to extract client descriptors and perform the one-shot clustering. After the clustering,
each client receives only the assigned cluster model, which its local model will be aggregated with other client models
in the same clusters. The training continues until the end. 
"""

# Libraries
import json
import copy
import time
import torch
import argparse
import pickle
import math
import numpy as np
from functools import reduce
from scipy.spatial.distance import euclidean, cosine
from logging import WARNING
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import List, Tuple, Union, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator # type: ignore

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import public.config as cfg
import public.utils as utils
import public.models as models
from modified_flwr.server import Server
from modified_flwr import app


import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.common.logger import log
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitRes,
    FitIns,
    Parameters,
    Scalar,
    Metrics,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
)

MAX_LATENT_SPACE = 2

# TODO DARIO
# WHAT IF: we introduced latent space descriptors per class, i.e., selecting only one class, calculating the mean latent
# space on it, reducing dim, and then same for others. Creating something like [metrics, latent_class1, latent_class2..]

class client_descr_scaling:
    def __init__(self, 
                 scaling_method: int = 1, 
                 scaler = None, # MinMaxScaler() or StandardScaler()
                 *args,
                 **kwargs):
        self.scaling_method = scaling_method
        self.scaler = scaler
        self.scalers = None
        self.fitted = False 
        if cfg.selected_descriptors == 'Px':
            self.descriptors_dim = [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors
            self.num_scalers = cfg.n_latent_space_descriptors
        elif cfg.selected_descriptors == 'Py':
            self.descriptors_dim = [cfg.len_metric_descriptor] * cfg.n_metrics_descriptors 
            self.num_scalers = cfg.n_metrics_descriptors
        elif cfg.selected_descriptors == 'Px_cond':
            self.descriptors_dim = [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors * 2
            self.num_scalers = cfg.n_latent_space_descriptors * 2
        elif cfg.selected_descriptors == 'Pxy_cond':
            self.descriptors_dim = [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors * 2 + [cfg.len_metric_descriptor] * cfg.n_metrics_descriptors
            self.num_scalers = cfg.n_latent_space_descriptors * 2 + cfg.n_metrics_descriptors
        elif cfg.selected_descriptors == 'Px_label_long':
            self.descriptors_dim = [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors * (cfg.n_classes + 1)
            self.num_scalers = cfg.n_latent_space_descriptors * (cfg.n_classes + 1)
        elif cfg.selected_descriptors == 'Px_label_short':
            self.descriptors_dim = [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors * 2
            self.num_scalers = cfg.n_latent_space_descriptors * 2
        else:
            self.descriptors_dim = [cfg.len_metric_descriptor] * cfg.n_metrics_descriptors + [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors
            self.num_scalers = cfg.n_metrics_descriptors + cfg.n_latent_space_descriptors

        print(f"n scalers: {self.num_scalers} - desc dim {self.descriptors_dim}")

    def scale(self, client_descr: np.ndarray = None) -> np.ndarray:
        # Normalize by group of descriptors
        if self.scaling_method == 1:
            if self.scalers is None:
                self.scalers = [copy.deepcopy(self.scaler) for _ in range(self.num_scalers)]
                self.dim = client_descr.shape[1]
             
            if self.fitted:
                if client_descr.shape[1] != self.dim:
                    raise ValueError("Client descriptors dimension mismatch!")
                scaled_client_descr = np.zeros(client_descr.shape)
                start_idx = 0
                for i, (scaler, descr_dim) in enumerate(zip(self.scalers, self.descriptors_dim)):
                    end_idx = start_idx + descr_dim
                    single_client_descr = client_descr[:, start_idx:end_idx]
                    scaled_client_descr[:, start_idx:end_idx] = scaler.transform(
                        single_client_descr.reshape(-1, 1)).reshape(single_client_descr.shape)
                    start_idx = end_idx
            else:
                self.fitted = True
                scaled_client_descr = np.zeros(client_descr.shape)
                start_idx = 0
                for i, (scaler, descr_dim) in enumerate(zip(self.scalers, self.descriptors_dim)):
                    end_idx = start_idx + descr_dim
                    single_client_descr = client_descr[:, start_idx:end_idx]
                    scaled_client_descr[:, start_idx:end_idx] = scaler.fit_transform(
                        single_client_descr.reshape(-1, 1)).reshape(single_client_descr.shape)
                    start_idx = end_idx
                
            return scaled_client_descr
        
        elif self.scaling_method == 2:
            # TODO weighted scaling
            return None
        
        elif self.scaling_method == 3:
            # No scaling
            return client_descr
        
        else:
            print("Invalid scaling method!")
            return None


# Config_client
def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "local_epochs": cfg.local_epochs,
        "tot_rounds": cfg.n_rounds,
        "extract_descriptors": False, 
        "min_latent_space": 0,
        "max_latent_space": MAX_LATENT_SPACE,
        "fedavg": True
    }
    return config


# Custom weighted average function
# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     # validities = [num_examples * m["validity"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]
#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Filter out entries where the accuracy is NaN
    valid_metrics = [(num_examples, m) for num_examples, m in metrics if not math.isnan(m["accuracy"])]
    
    # If no valid metrics remain, return NaN as the overall accuracy
    if not valid_metrics:
        return {"accuracy": float('nan')}
    
    # Compute the weighted sum of accuracies for valid metrics
    weighted_sum = sum(num_examples * m["accuracy"] for num_examples, m in valid_metrics)
    total_examples = sum(num_examples for num_examples, _ in valid_metrics)
    
    # Return the weighted average
    return {"accuracy": weighted_sum / total_examples}


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


def weighted_aggregate(results: List[Tuple[NDArrays, int]], client_weight_distance:NDArrays) -> NDArrays:
    """Compute weighted average with distances."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples * w for (_, num_examples), w in zip(results, client_weight_distance)])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples * w for layer in weights] for (weights, num_examples), w in zip(results, client_weight_distance)
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


# Custom strategy to save model after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, path, descriptors_scaler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model # used for saving checkpoints
        self.path = path # saving model path
        self.descriptors_scaler = descriptors_scaler # used for scaling client descriptors

        self.aggregated_client_parameters = {} # [cluster_label] = model parameters
        self.client_descriptors = {} # [client_id] = descriptors
        self.aggregated_parameters_global = None
        self.fedavg = True # True: Fedavg training, False: personalized clustering # 0: not started, 1: to cluster, 2: done
        # self.starting_round = True
        self.save_client_id_cid = True
        self.parent_client_descrs = None
        # ZZZ
        self.accuracy_trend = [] # accuracy trend for clustering


    # Override configure_fit method to add custom configuration
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager, 
        descriptor_extraction: bool = False
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)      # Config sent to clients during training 
            if self.fedavg == False:
                config["fedavg"] = False
                if descriptor_extraction == True:
                    config["extract_descriptors"] = True
                else:
                    config["extract_descriptors"] = False
            else:
                config["fedavg"] = True
                config["extract_descriptors"] = False
            
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # If still fedavg
        if self.fedavg:
            fit_ins = FitIns(parameters, config)
            return [(client, fit_ins) for client in clients]
        
        else:
            if descriptor_extraction:
                # do not send model to clients - they will use the global model
                # i send this to not raise an error
                fit_ins = FitIns(parameters, config)
                return [(client, fit_ins) for client in clients]
            else:
                # send the personalized clustered model 
                return [(client, 
                         FitIns(self.aggregated_client_parameters[client.cid], config)) for client in clients]


    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        descriptor_extraction: bool = False,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        
        # Fedavg
        if self.fedavg:
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
            self.aggregated_parameters_global = ndarrays_to_parameters(aggregate(weights_results))   # Global aggregation - traditional - no clustering
            
            # Aggregate custom metrics if aggregation fn was provided   NO FIT METRICS AGGREGATION FN PROVIDED - SKIPPED FOR NOW
            aggregated_metrics = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")

            # Save model
            if self.aggregated_parameters_global is not None:

                print(f"Saving round {server_round} aggregated_parameters...")
                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(self.aggregated_parameters_global)
                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)
                # Save the model. TODO: save only best accuracy model and loss model
                torch.save(self.model.state_dict(), f"checkpoints/{self.path}/{cfg.non_iid_type}_global_model.pth")
            
            return self.aggregated_parameters_global, aggregated_metrics
        
        else:
            if descriptor_extraction:
                # Extract & scale client descriptors and self-assigned client ids, FLWR cids
                # client_descr, client_id_plot, client_cid_list  = [], [], []
                client_descr = []
                client_id_cid = {}
                for proxy, res in results:
                    if cfg.extended_descriptors:
                        if cfg.selected_descriptors == "Pxy":
                            if res.metrics["cid"] == 1:
                                print(f"\033[91mClustering using extended Pxy descriptors\033[0m")
                            client_descr.append(json.loads(res.metrics["loss_pc_mean"]) + \
                                                json.loads(res.metrics["loss_pc_std"]) + \
                                                json.loads(res.metrics["latent_space_mean"]) + \
                                                json.loads(res.metrics["latent_space_std"]))
                        elif cfg.selected_descriptors == "Py":
                            if res.metrics["cid"] == 1:
                                print(f"\033[91mClustering using extended Py descriptors\033[0m")
                            client_descr.append(json.loads(res.metrics["loss_pc_mean"]) + \
                                                json.loads(res.metrics["loss_pc_std"]))
                        elif cfg.selected_descriptors == "Px":
                            if res.metrics["cid"] == 1:
                                print(f"\033[91mClustering using extended Px descriptors\033[0m")
                            client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                                json.loads(res.metrics["latent_space_std"]))
                        elif cfg.selected_descriptors == "Px_cond":
                            if res.metrics["cid"] == 1:
                                print(f"\033[91mClustering using extended Px_cond descriptors\033[0m")
                            client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                                json.loads(res.metrics["latent_space_std"]) + \
                                                json.loads(res.metrics["latent_space_cond_mean"]) + \
                                                json.loads(res.metrics["latent_space_cond_std"]))
                        elif cfg.selected_descriptors == "Pxy_cond":
                            if res.metrics["cid"] == 1:
                                print(f"\033[91mClustering using extended Pxy_cond descriptors\033[0m")
                            client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                                json.loads(res.metrics["latent_space_std"]) + \
                                                json.loads(res.metrics["latent_space_cond_mean"]) + \
                                                json.loads(res.metrics["latent_space_cond_std"]) + \
                                                json.loads(res.metrics["loss_pc_mean"]) + \
                                                json.loads(res.metrics["loss_pc_std"]))
                        elif cfg.selected_descriptors == "Px_label_long":
                            if res.metrics["cid"] == 1:
                                print(f"\033[91mClustering using extended Px_label_long descriptors\033[0m")
                            client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                                json.loads(res.metrics["latent_space_std"]) + \
                                                json.loads(res.metrics["latent_space_mean_by_label"]) + \
                                                json.loads(res.metrics["latent_space_std_by_label"]))
                        elif cfg.selected_descriptors == "Px_label_short":
                            if res.metrics["cid"] == 1:
                                print(f"\033[91mClustering using extended Px_label_short descriptors\033[0m")
                            client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                                json.loads(res.metrics["latent_space_std"]) + \
                                                json.loads(res.metrics["latent_space_mean_by_label"]) + \
                                                json.loads(res.metrics["latent_space_std_by_label"]))
                    else:    
                        if cfg.selected_descriptors == "Pxy":
                            if res.metrics["cid"] == 1:
                                print(f"\033[91mClustering using basic Pxy descriptors\033[0m")
                            client_descr.append(json.loads(res.metrics["loss_pc_mean"]) + \
                                                json.loads(res.metrics["latent_space_mean"]))
                        elif cfg.selected_descriptors == "Py":
                            if res.metrics["cid"] == 1:
                                print(f"\033[91mClustering using basic Py descriptors\033[0m")
                            client_descr.append(json.loads(res.metrics["loss_pc_std"]))
                        elif cfg.selected_descriptors == "Px":
                            if res.metrics["cid"] == 1:
                                print(f"\033[91mClustering using basic Px descriptors\033[0m")
                            client_descr.append(json.loads(res.metrics["latent_space_mean"]))
                        elif cfg.selected_descriptors == "Px_cond":
                            if res.metrics["cid"] == 1:
                                print(f"\033[91mClustering using basic Px_cond descriptors\033[0m")
                            client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                                json.loads(res.metrics["latent_space_cond_mean"]))
                        elif cfg.selected_descriptors == "Pxy_cond":
                            if res.metrics["cid"] == 1:
                                print(f"\033[91mClustering using basic Pxy_cond descriptors\033[0m")
                            client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                                json.loads(res.metrics["latent_space_cond_mean"]) + \
                                                json.loads(res.metrics["loss_pc_mean"]))
                        elif cfg.selected_descriptors == "Px_label_long":
                            if res.metrics["cid"] == 1:
                                print(f"\033[91mClustering using basic Px_label_long descriptors\033[0m")
                            client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                                json.loads(res.metrics["latent_space_mean_by_label"]))
                        elif cfg.selected_descriptors == "Px_label_short":
                            if res.metrics["cid"] == 1:
                                print(f"\033[91mClustering using basic Px_label_short descriptors\033[0m")
                            client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                                json.loads(res.metrics["latent_space_mean_by_label"]))                        
                            
                    # client_id_plot.append(res.metrics["cid"])
                    # client_cid_list.append(proxy.cid)
                    client_id_cid[res.metrics["cid"]] = proxy.cid

                # scaling
                client_descr = self.descriptors_scaler.scale(np.array(client_descr))
                print(f"\033[91mRound {server_round} - Shape descriptors {client_descr.shape}\033[0m")
                # print(f"\033[91mRound {server_round} - Scaled client descriptors {client_descr}\033[0m")
                
                # Apply PCA to reduce the data to 2D for visualization
                # X_reduced = PCA(n_components=2).fit_transform(client_descr)

                # temp save
                # save descriptors and cid list
                # np.save(f'results/client_descr.npy', client_descr)
                # np.save(f'results/client_id_plot.npy', client_id_plot)
                if self.save_client_id_cid:
                    self.save_client_id_cid = False
                    with open(f'results/{self.path}/client_cid_list.pkl', 'wb') as f:
                        pickle.dump(client_id_cid, f)

                def emphasize_high_values(values, power=5):
                    """Increase the weight of higher values and reduce the lower ones."""
                    # check if numpy array otherwise convert
                    if not isinstance(values, np.ndarray):
                        values = np.array(values)
                        
                    adjusted = values ** power
                    return adjusted / adjusted.sum()

                def cal_weights(
                    parent_client_descrs: List[np.ndarray] = None,
                    cur_client_descrs: List[np.ndarray] = None,
                    dis_func: str = "euclidean" # "euclidean", "cosine"
                ) -> List[np.ndarray]:
                    '''
                    Calculate the weights for the weighted aggregation

                    Args:
                        starting_round: bool - if it is the first round
                        parent_client_descr: List[np.ndarray] - descriptors of the parent clients (linked to one model)
                        cur_client_descr: List[np.ndarray] - descriptors of the current clients

                    Returns:
                        List[np.ndarray]: weights for the weighted aggregation (normalized)
                    '''
                    # starting round, one fedavg model, still do fedavg
                    if parent_client_descrs is None:
                        return [np.full(cfg.n_clients, 1 / cfg.n_clients) for _ in range(cfg.n_clients)]
                    else:
                            if dis_func == "euclidean":
                                distance_fn = euclidean
                            elif dis_func == "cosine":
                                distance_fn = cosine
                            else:
                                raise ValueError("dis_func must be 'euclidean' or 'cosine'.")
                    weight_matrix = []
                    
                    i=0
                    for cur_descr in cur_client_descrs:
                        weights = []
                        for par_descr in parent_client_descrs:
                            dist = distance_fn(cur_descr, par_descr)
                            if server_round == 7 and i == 0:
                                print(f"Distance: {dist}")
                                
                            weights.append(1.0 / (dist + 1e-8))
                        i+=1
                        weights = np.array(weights)
                        # save the weights for the current client
                        # np.save(f'results/{self.path}/weights_{i}.npy', weights)
                        # if cfg.softmax:
                        if False:
                            weights = weights / weights.sum()
                            normalized_weights = emphasize_high_values(weights, power=4)
                        elif False:
                            normalized_weights = weights / weights.sum()
                            # set to zero if the value is lower than 0.005
                            normalized_weights[normalized_weights < 0.04] = 0
                            print(f"Normalized weights: {normalized_weights}")
                            normalized_weights = normalized_weights / normalized_weights.sum()
                        else:
                            normalized_weights = weights / weights.sum()
                        weight_matrix.append(normalized_weights)

                    return weight_matrix   

                client_distances = cal_weights(
                    parent_client_descrs=self.parent_client_descrs,
                    cur_client_descrs=client_descr,
                    dis_func=cfg.distance_function
                )
                # update parent client descriptors
                self.parent_client_descrs = client_descr

                # save the weights for visualization
                if cfg.distance_visualization:
                    client_distances_matrix = np.array(client_distances)
                    np.save(f'results/{self.path}/client_distances_matrix_round_{server_round}.npy', client_distances_matrix)
                    with open(f'results/{self.path}/client_cid_list_round_{server_round}.pkl', 'wb') as f:
                        pickle.dump(client_id_cid, f)

                # client_distances = [np.ones(cfg.n_clients) for _ in range(cfg.n_clients)]
                print(f"\033[91mRound {server_round} - Client distances: {client_distances}\033[0m")

                # Aggregation
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

                cur_round_cids = [proxy.cid for proxy, _ in results]
        
                # Aggregate each cluster
                for n, (cid, w) in enumerate(zip(cur_round_cids, client_distances)):
                    # print(f"Aggregating cluster {cid}...")
                    self.aggregated_client_parameters[cid] = ndarrays_to_parameters(weighted_aggregate(weights_results, w))
                    self.client_descriptors[cid] = client_descr[n]
            
                # if last round, save the models and the respective descriptors (here i'm saving the just aggregated model)
                if server_round == cfg.n_rounds:
                    # for cid, params in self.aggregated_client_parameters.items():
                    #     print(f"Saving round {server_round} aggregated_client_parameters_{cid}...")
                    #     # Convert `Parameters` to `List[np.ndarray]`
                    #     aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(params)
                    #     # Convert `List[np.ndarray]` to PyTorch`state_dict`
                    #     params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
                    #     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                    #     self.model.load_state_dict(state_dict, strict=True)
                    #     # Overwrite the model
                    #     torch.save(self.model.state_dict(), f"checkpoints/{self.path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_cid_{cid}.pth")
                        
                    # save client descriptors
                    with open(f'results/{self.path}/client_descriptors.pkl', 'wb') as f:
                        pickle.dump(self.client_descriptors, f)
                    
                # i pass this to not raise an error
                return self.aggregated_parameters_global, {}

            else:
                # do not do nothing - no aggregation 
                # pass this to not raise an error
                
                # we can save also here the model (after local training)
                # Aggregation
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

                cur_round_cids = [proxy.cid for proxy, _ in results]

                # Save trained client models
                trained_models = {}
                for n, cid in enumerate(cur_round_cids):
                    trained_models[cid] = ndarrays_to_parameters(weights_results[n][0])
            
                # if last round, save the models and the respective descriptors (here i'm saving the just aggregated model)
                if server_round == cfg.n_rounds:
                    for cid, params in trained_models.items():
                        # print(f"Saving round {server_round} trained_model_{cid}...")
                        # Convert `Parameters` to `List[np.ndarray]`
                        aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(params)
                        # Convert `List[np.ndarray]` to PyTorch`state_dict`
                        params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
                        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                        self.model.load_state_dict(state_dict, strict=True)
                        # Overwrite the model
                        torch.save(self.model.state_dict(), f"checkpoints/{self.path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_cid_{cid}_trained.pth")
                
                return self.aggregated_parameters_global, {}

   
    # Override configure_evaluate method to add custom configuration
    def configure_evaluate(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
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
        
        # If still fedavg
        if self.fedavg:
            fit_ins = FitIns(parameters, config)
            return [(client, fit_ins) for client in clients]
        
        else:
            # send the personalized clustered model 
            return [(client, 
                        FitIns(self.aggregated_client_parameters[client.cid], config)) for client in clients]

 
    # Override
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

        # Clustering requirements detection
        print(f"\033[93mRound {server_round} - Aggregated loss: {loss_aggregated} - Aggregated metrics: {metrics_aggregated}\033[0m")
        self.accuracy_trend.append(metrics_aggregated["accuracy"])
        
        if self.fedavg == True: 
            # Update the max_latent_space for the next round
            max_client_latent_space = max([res.metrics["max_latent_space"] for _, res in results])
            global MAX_LATENT_SPACE 
            MAX_LATENT_SPACE = 1.02 * max_client_latent_space 
            
        # Cluster requirements check
        if self.fedavg == True:
            # if server_round >= 3:
            #     d_accuracy = np.diff(self.accuracy_trend)
            #     if any(d_accuracy < cfg.th_round):
            #         print(f"\033[93mRound {server_round} - Threshold reached \033[0m")
            #         self.cluster_status = 1
            #     elif server_round >= 0.8 * cfg.n_rounds:
            #         print(f"\033[93mRound {server_round} - Threshold not reached, but 80% of rounds done\033[0m")
            #         self.cluster_status = 1
            if server_round > 4:
                self.fedavg = False
                print(f"\033[93mRound {server_round} - Threshold reached - Starting personalized aggregation  \033[0m")
            else:
                print(f"\033[93mRound {server_round} - No need to extract descriptors yet - training in FedAvg\033[0m")

        return loss_aggregated, metrics_aggregated

















# Main
def main() -> None:
    # Get arguments
    parser = argparse.ArgumentParser(description='Clustered Federated Learning - Server')
    parser.add_argument('--fold', type=int, default=0, help='Fold number of the cross-validation')
    args = parser.parse_args()

    utils.set_seed(cfg.random_seed + args.fold)
    start_time = time.time()
    exp_path = utils.create_folders()
    device = utils.check_gpu()
    in_channels = utils.get_in_channels()
    model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                          input_size=cfg.input_size).to(device)
    descriptors_scaler = client_descr_scaling(scaling_method=cfg.cfl_oneshot_CLIENT_SCALING_METHOD,
                                              scaler=MinMaxScaler(),
                                              )
    
    # Define strategy
    strategy = SaveModelStrategy(
        # self defined
        model=model,
        path=exp_path,
        descriptors_scaler=descriptors_scaler,
        # super
        min_fit_clients=cfg.n_clients, # always all training
        min_evaluate_clients=cfg.n_clients, # always all evaluating
        min_available_clients=cfg.n_clients, # always all available
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
    )

    # Start Flower server for three rounds of federated learning
    history = app.start_server(
        server=Server(client_manager=SimpleClientManager(), strategy=strategy),
        server_address=f"{cfg.ip}:{cfg.port}",   # 0.0.0.0 listens to all available interfaces
        config=fl.server.ServerConfig(num_rounds=cfg.n_rounds),
        # strategy=strategy,
    )

    # Convert history to list
    loss = [k[1] for k in history.losses_distributed]
    accuracy = [k[1] for k in history.metrics_distributed['accuracy']]

    # Save loss and accuracy to a file
    print(f"Saving metrics to as .json in histories folder...")
    with open(f'histories/{exp_path}/distributed_metrics_{args.fold}.json', 'w') as f:
        json.dump({'loss': loss, 'accuracy': accuracy}, f)

    # Plot client training loss and accuracy
    utils.plot_all_clients_metrics(fold=args.fold)

    # Plots and Evaluation the model on the client datasets
    best_loss_round, best_acc_round = utils.plot_loss_and_accuracy(loss, accuracy, show=False, fold=args.fold)
    
    # Read cluster centroids from json - for test-time inference
    client_descriptors = np.load(f'results/{exp_path}/client_descriptors.pkl', allow_pickle=True)
    if cfg.selected_descriptors == "Pxy":
        client_descriptors = {cid: descriptor[cfg.n_metrics_descriptors*cfg.len_metric_descriptor:] for cid, descriptor in client_descriptors.items()} # only latent space
        # print(f"\033[93mCluster centroids: {client_descriptors}\033[0m\n")
    elif cfg.selected_descriptors == "Px":
        # print(f"\033[93mCluster centroids: {client_descriptors}\033[0m\n") # only latent space
        pass
    elif cfg.selected_descriptors == "Py":
        raise ValueError("You cannot use Py at inference, dummy guy! I will read cluster assignement during training for inference")
    elif cfg.selected_descriptors in ["Px_cond", "Pxy_cond", "Px_label_long", "Px_label_short"]:
        if not cfg.non_iid_type == "Py_x":
            client_descriptors = {cid: descriptor[:cfg.n_latent_space_descriptors*cfg.len_latent_space_descriptor] for cid, descriptor in client_descriptors.items()}
        # print(f"\033[93mCluster centroids: {client_descriptors}\033[0m\n") # only latent space
    else:
        raise ValueError("Invalid selected_descriptors")
    
    # Load global model for evaluation (descriptor extraction)
    evaluation_model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                          input_size=cfg.input_size).to(device)
    evaluation_model.load_state_dict(torch.load(f"checkpoints/{exp_path}/{cfg.non_iid_type}_global_model.pth", weights_only=False))
    
    # Evaluate the model on the client datasets 
    print(f"\n\033[94mEuclidean Distance \033[0m")   
    losses, accuracies = [], []
    losses_known, accuracies_known = [], []
    for client_id in range(cfg.n_clients):
        test_x, test_y = [], []
        if not cfg.training_drifting:
            cur_data = np.load(f'../data/cur_datasets/client_{client_id}.npy', allow_pickle=True).item()
            cur_data['test_features'] = torch.tensor(cur_data['test_features'], dtype=torch.float32)
            cur_data['test_labels'] = torch.tensor(cur_data['test_labels'], dtype=torch.int64)
            if not cfg.dataset_name == "CheXpert":
                test_x = cur_data['test_features'] if in_channels == 3 else cur_data['test_features'].unsqueeze(1)
            else:
                test_x = cur_data['test_features']
            test_y = cur_data['test_labels']
        else:
            cur_data = np.load(f'../data/cur_datasets/client_{client_id}_round_-1.npy', allow_pickle=True).item()
            cur_data['features'] = torch.tensor(cur_data['features'], dtype=torch.float32)
            cur_data['labels'] = torch.tensor(cur_data['labels'], dtype=torch.int64)
            if not cfg.dataset_name == "CheXpert":
                test_x = cur_data['features'] if in_channels == 3 else cur_data['features'].unsqueeze(1)
            else:
                test_x = cur_data['features']
            test_y = cur_data['labels']
        
        # Create test dataset and loader
        test_dataset = models.CombinedDataset(test_x, test_y, transform=None)
        test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)
    
        # --- Test-time inference: check closest cluster ---
        # Extract descriptors, scaling
        descriptors = models.ModelEvaluator(test_loader=test_loader, device=device).extract_descriptors_inference(
                                                    model=evaluation_model, max_latent_space=MAX_LATENT_SPACE)
        
        if cfg.selected_descriptors == "Pxy":
            descriptors = descriptors_scaler.scale(descriptors.reshape(1,-1))
            descriptors = descriptors[:, cfg.n_metrics_descriptors*cfg.len_metric_descriptor:][0] # only latent space
        elif cfg.selected_descriptors == "Px":
            descriptors = descriptors[cfg.n_metrics_descriptors*cfg.len_metric_descriptor:] # only latent space 
            descriptors = descriptors_scaler.scale(descriptors.reshape(1,-1))[0]
        elif cfg.selected_descriptors in ["Px_cond", "Pxy_cond", "Px_label_long", "Px_label_short"]:
            descriptors = descriptors_scaler.scale(descriptors.reshape(1,-1))
            descriptors = descriptors[:, :cfg.n_latent_space_descriptors*cfg.len_latent_space_descriptor][0] # only latent space
        else:
            raise ValueError("Invalid selected_descriptors")
        
        # if Py_x condition few labels are needed
        if cfg.non_iid_type == "Py_x":
            # -- Real-Association with few labels
            if cfg.n_test_sample_per_class == -1:
                test_loader_label = test_loader
            else: 
                # sample n_test_sample_per_class from test dataset
                unique_labels = np.unique(test_y)
                sampled_indices = []
                for lbl in unique_labels:
                    indices = [i for i, yval in enumerate(test_y) if yval == lbl]
                    if len(indices) > cfg.n_test_sample_per_class:
                        sampled_idx = np.random.choice(indices, cfg.n_test_sample_per_class, replace=False)
                    else:
                        sampled_idx = indices
                    sampled_indices.extend(sampled_idx)

                sampled_test_x = [test_x[i] for i in sampled_indices]
                sampled_test_y = [test_y[i] for i in sampled_indices]
                sampled_test_dataset = models.CombinedDataset(sampled_test_x, sampled_test_y, transform=None)

                # Create the new test data loader
                test_loader_label = DataLoader(sampled_test_dataset, batch_size=cfg.n_test_sample_per_class, shuffle=False)

            # Extract descriptors, scaling
            descriptors_label = models.ModelEvaluator(test_loader=test_loader_label, device=device).extract_descriptors_inference(
                                                        model=evaluation_model, max_latent_space=MAX_LATENT_SPACE)
            descriptors_label = descriptors_scaler.scale(descriptors_label.reshape(1,-1))
            # Combine second part of this descriptor to the Px one (comment this line to use only descriptors_label[0])
            descriptors_label = descriptors_label[:, cfg.n_latent_space_descriptors*cfg.len_latent_space_descriptor:][0] # only latent space
            # Concatenation (comment this line to use only descriptors_label[0])
            descriptors = np.concatenate([descriptors, descriptors_label])
            # using only this descriptors (uncomment this line to use only descriptors_label[0])
            # descriptors = np.array(descriptors_label[0])
            
            ## -- Known-Association evaluation --
            # load client id - cid reference
            known_client_cid = np.load(f'results/{exp_path}/client_cid_list.pkl', allow_pickle=True)[client_id]
            known_model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                            input_size=cfg.input_size).to(device)
            known_model.load_state_dict(torch.load(f"checkpoints/{exp_path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_cid_{known_client_cid}_trained.pth", weights_only=False))
            # Evaluate
            loss_known, accuracy_known = models.simple_test(known_model, device, test_loader)
            print(f"\033[93mKnown: Client {client_id} - Test Loss: {loss_known:.3f}, Test Accuracy: {accuracy_known*100:.2f} - Associciate model cid: {known_client_cid}\033[0m")
            accuracies_known.append(accuracy_known)
            losses_known.append(loss_known) 

        # Find the closest model to the client: TODO: use the right distance function
        # client_model_cid_old = min(client_descriptors, key=lambda cid: np.linalg.norm(descriptors - client_descriptors[cid]))
        if cfg.distance_function == "euclidean":
            distance_fn = euclidean
        elif cfg.distance_function == "cosine":
            distance_fn = cosine
        else:
            raise ValueError("dis_func must be 'euclidean' or 'cosine'.")
        # descriptors = np.array(descriptors[0])
        client_model_cid = min(client_descriptors, key=lambda cid: euclidean(descriptors, client_descriptors[cid]))
        # print(f"\033[93mClient {client_id} - Old closest centroid: {client_model_cid_old}, New closest centroid: {client_model_cid}\033[0m")

        # Load respective cluster model
        test_client_model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                        input_size=cfg.input_size).to(device)
        test_client_model.load_state_dict(torch.load(f"checkpoints/{exp_path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_cid_{client_model_cid}_trained.pth", weights_only=False))

        # Evaluate
        loss_test, accuracy_test = models.simple_test(test_client_model, device, test_loader)
        print(f"\033[93mClient {client_id} - Test Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f} - Associciate model cid: {client_model_cid}\033[0m")
        accuracies.append(accuracy_test)
        losses.append(loss_test)

    # print average loss and accuracy
    print(f"\n\033[93mAverage Loss: {np.nanmean(losses):.3f}, Average Accuracy: {np.nanmean(accuracies)*100:.2f}\033[0m")
    print(f"\033[93mAverage Loss (known): {np.nanmean(losses_known):.3f}, Average Accuracy (known): {np.nanmean(accuracies_known)*100:.2f}\033[0m")
    print(f"\033[90mTraining time: {round((time.time() - start_time)/60, 2)} minutes \n\n\n\033[0m")
  
  
  
    # ---- temp with different distance function ----
    # Evaluate the model on the client datasets  
    print(f"\n\033[94mCosine Distance \033[0m")  
    losses_cosine, accuracies_cosine = [], []
    losses_known_cosine, accuracies_known_cosine = [], []
    for client_id in range(cfg.n_clients):
        test_x, test_y = [], []
        if not cfg.training_drifting:
            cur_data = np.load(f'../data/cur_datasets/client_{client_id}.npy', allow_pickle=True).item()
            if not cfg.dataset_name == "CheXpert":
                test_x = cur_data['test_features'] if in_channels == 3 else cur_data['test_features'].unsqueeze(1)
            else:
                test_x = cur_data['test_features']
            test_y = cur_data['test_labels']
        else:
            cur_data = np.load(f'../data/cur_datasets/client_{client_id}_round_-1.npy', allow_pickle=True).item()
            if not cfg.dataset_name == "CheXpert":
                test_x = cur_data['features'] if in_channels == 3 else cur_data['features'].unsqueeze(1)
            else:
                test_x = cur_data['features']
            test_y = cur_data['labels']
        
        # Create test dataset and loader
        test_dataset = models.CombinedDataset(test_x, test_y, transform=None)
        test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)
    
        # --- Test-time inference: check closest cluster ---
        # Extract descriptors, scaling
        descriptors = models.ModelEvaluator(test_loader=test_loader, device=device).extract_descriptors_inference(
                                                    model=evaluation_model, max_latent_space=MAX_LATENT_SPACE)
        
        if cfg.selected_descriptors == "Pxy":
            descriptors = descriptors_scaler.scale(descriptors.reshape(1,-1))
            descriptors = descriptors[:, cfg.n_metrics_descriptors*cfg.len_metric_descriptor:][0] # only latent space
        elif cfg.selected_descriptors == "Px":
            descriptors = descriptors[cfg.n_metrics_descriptors*cfg.len_metric_descriptor:] # only latent space 
            descriptors = descriptors_scaler.scale(descriptors.reshape(1,-1))[0]
        elif cfg.selected_descriptors in ["Px_cond", "Pxy_cond", "Px_label_long", "Px_label_short"]:
            descriptors = descriptors_scaler.scale(descriptors.reshape(1,-1))
            descriptors = descriptors[:, :cfg.n_latent_space_descriptors*cfg.len_latent_space_descriptor][0] # only latent space
        else:
            raise ValueError("Invalid selected_descriptors")
        
        # if Py_x condition few labels are needed
        if cfg.non_iid_type == "Py_x":
            # -- Real-Association with few labels
            if cfg.n_test_sample_per_class == -1:
                test_loader_label = test_loader
            else: 
                # sample n_test_sample_per_class from test dataset
                unique_labels = np.unique(test_y)
                sampled_indices = []
                for lbl in unique_labels:
                    indices = [i for i, yval in enumerate(test_y) if yval == lbl]
                    if len(indices) > cfg.n_test_sample_per_class:
                        sampled_idx = np.random.choice(indices, cfg.n_test_sample_per_class, replace=False)
                    else:
                        sampled_idx = indices
                    sampled_indices.extend(sampled_idx)

                sampled_test_x = [test_x[i] for i in sampled_indices]
                sampled_test_y = [test_y[i] for i in sampled_indices]
                sampled_test_dataset = models.CombinedDataset(sampled_test_x, sampled_test_y, transform=None)

                # Create the new test data loader
                test_loader_label = DataLoader(sampled_test_dataset, batch_size=cfg.n_test_sample_per_class, shuffle=False)

            # Extract descriptors, scaling
            descriptors_label = models.ModelEvaluator(test_loader=test_loader_label, device=device).extract_descriptors_inference(
                                                        model=evaluation_model, max_latent_space=MAX_LATENT_SPACE)
            descriptors_label = descriptors_scaler.scale(descriptors_label.reshape(1,-1))
            # Combine second part of this descriptor to the Px one (comment this line to use only descriptors_label[0])
            descriptors_label = descriptors_label[:, cfg.n_latent_space_descriptors*cfg.len_latent_space_descriptor:][0] # only latent space
            # Concatenation (comment this line to use only descriptors_label[0])
            descriptors = np.concatenate([descriptors, descriptors_label])
            # using only this descriptors (uncomment this line to use only descriptors_label[0])
            # descriptors = np.array(descriptors_label[0])
            
            ## -- Known-Association evaluation --
            # load client id - cid reference
            known_client_cid = np.load(f'results/{exp_path}/client_cid_list.pkl', allow_pickle=True)[client_id]
            known_model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                            input_size=cfg.input_size).to(device)
            known_model.load_state_dict(torch.load(f"checkpoints/{exp_path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_cid_{known_client_cid}_trained.pth", weights_only=False))
            # Evaluate
            loss_known, accuracy_known = models.simple_test(known_model, device, test_loader)
            print(f"\033[93mKnown: Client {client_id} - Test Loss: {loss_known:.3f}, Test Accuracy: {accuracy_known*100:.2f} - Associciate model cid: {known_client_cid}\033[0m")
            accuracies_known_cosine.append(accuracy_known)
            losses_known_cosine.append(loss_known) 

        # Find the closest model to the client: TODO: use the right distance function
        # client_model_cid_old = min(client_descriptors, key=lambda cid: np.linalg.norm(descriptors - client_descriptors[cid]))
        if cfg.distance_function == "euclidean":
            distance_fn = euclidean
        elif cfg.distance_function == "cosine":
            distance_fn = cosine
        else:
            raise ValueError("dis_func must be 'euclidean' or 'cosine'.")
        # descriptors = np.array(descriptors[0])
        client_model_cid = min(client_descriptors, key=lambda cid: cosine(descriptors, client_descriptors[cid]))
        # print(f"\033[93mClient {client_id} - Old closest centroid: {client_model_cid_old}, New closest centroid: {client_model_cid}\033[0m")

        # Load respective cluster model
        test_client_model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                        input_size=cfg.input_size).to(device)
        test_client_model.load_state_dict(torch.load(f"checkpoints/{exp_path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_cid_{client_model_cid}_trained.pth", weights_only=False))

        # Evaluate
        loss_test, accuracy_test = models.simple_test(test_client_model, device, test_loader)
        print(f"\033[93mClient {client_id} - Test Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f} - Associciate model cid: {client_model_cid}\033[0m")
        accuracies_cosine.append(accuracy_test)
        losses_cosine.append(loss_test)

    # print average loss and accuracy
    print(f"\n\033[93mAverage Loss: {np.nanmean(losses_cosine):.3f}, Average Accuracy: {np.nanmean(accuracies_cosine)*100:.2f}\033[0m")
    print(f"\033[93mAverage Loss (known): {np.nanmean(losses_known_cosine):.3f}, Average Accuracy (known): {np.nanmean(accuracies_known_cosine)*100:.2f}\033[0m")
    print(f"\033[90mTraining time: {round((time.time() - start_time)/60, 2)} minutes\033[0m")  
  
    # Save metrics as numpy array
    metrics = {
        "loss": losses,
        "accuracy": accuracies,
        "average_loss": np.nanmean(losses),
        "average_accuracy": np.nanmean(accuracies),
        "average_loss_cosine": np.nanmean(losses_cosine),
        "average_accuracy_cosine": np.nanmean(accuracies_cosine),
        "time": round((time.time() - start_time)/60, 2),
    }
    if cfg.non_iid_type == "Py_x":
        metrics["average_loss_known"] = np.nanmean(losses_known)
        metrics["average_accuracy_known"] = np.nanmean(accuracies_known)
        metrics["average_loss_known_cosine"] = np.nanmean(losses_known_cosine)
        metrics["average_accuracy_known_cosine"] = np.nanmean(accuracies_known_cosine)
    np.save(f'results/{exp_path}/test_metrics_fold_{args.fold}.npy', metrics)
    
    time.sleep(1)
    
if __name__ == "__main__":
    main()
