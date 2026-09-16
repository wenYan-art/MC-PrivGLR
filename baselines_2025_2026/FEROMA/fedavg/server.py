"""
CFL implementation of fedavg, server side.

Code to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
run the appopriate client code (client.py).
"""

from typing import List, Tuple, Union, Optional, Dict
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from logging import WARNING
from collections import OrderedDict
import json
import time
from functools import reduce
import math

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import public.config as cfg
import public.utils as utils
import public.models as models

import flwr as fl
from flwr.common import Parameters, Scalar, Metrics
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
)

# Config_client
def fit_config(
        server_round: int
    ) -> Dict[str, Scalar]:
    """
        Generate training configuration dict for each round.
    """
    config = {
        "current_round": server_round,
        "local_epochs": cfg.local_epochs,
        "tot_rounds": cfg.n_rounds,
        "min_latent_space": 0,
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

# Custom strategy to save model after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model  # used for saving checkpoints
        self.path = path # saving model path

    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
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
        aggregated_parameters_global = ndarrays_to_parameters(aggregate(weights_results))   # Global aggregation - traditional - no clustering
        
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
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters_global)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            # Save the model. TODO: save only best accuracy model and loss model
            torch.save(self.model.state_dict(), f"checkpoints/{self.path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_round_{server_round}.pth")
        
        return aggregated_parameters_global, aggregated_metrics

def main() -> None:
    # Get arguments
    parser = argparse.ArgumentParser(description='FedAvg - Server')
    parser.add_argument('--fold', type=int, default=0, help='Fold number of the cross-validation')
    args = parser.parse_args()

    utils.set_seed(cfg.random_seed + args.fold)
    start_time = time.time()
    exp_path = utils.create_folders()
    device = utils.check_gpu()
    in_channels = utils.get_in_channels()
    model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                          input_size=cfg.input_size).to(device)

    # Define strategy
    strategy = SaveModelStrategy(
        # self defined
        model=model,
        path=exp_path,
        # super
        min_fit_clients=cfg.n_clients, # always all training
        min_evaluate_clients=cfg.n_clients, # always all evaluating
        min_available_clients=cfg.n_clients, # always all available
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
    )

    # Start Flower server and (finish all training and evaluation)
    history = fl.server.start_server(
        server_address=f"{cfg.ip}:{cfg.port}",   # 0.0.0.0 listens to all available interfaces
        config=fl.server.ServerConfig(num_rounds=cfg.n_rounds),
        strategy=strategy,
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

    # Plots and Evaluation the model on the client datasets, (averaged)
    best_loss_round, best_acc_round = utils.plot_loss_and_accuracy(loss, accuracy, show=False)
    model.load_state_dict(torch.load(f"checkpoints/{exp_path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_round_{best_loss_round}.pth", weights_only=False))

    # Evaluate the model on the client datasets    
    losses, accuracies = [], []
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

        # Evaluate on client
        loss_test, accuracy_test = models.simple_test(model, device, test_loader)
        print(f"\033[93mClient {client_id} - Test Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f}\033[0m")
        accuracies.append(accuracy_test)
        losses.append(loss_test)
    
    # Averaged accuracy across clients   
    print(f"\n\033[93mAverage Test Loss: {np.nanmean(losses):.3f}, Average Test Accuracy: {np.nanmean(accuracies)*100:.2f}\033[0m\n")
    print(f"\033[90mTraining time: {round((time.time() - start_time)/60, 2)} minutes\033[0m")
    
    # Save metrics as numpy array
    metrics = {
        "loss": losses,
        "accuracy": accuracies,
        "average_loss": np.nanmean(losses),
        "average_accuracy": np.nanmean(accuracies),
        "time": round((time.time() - start_time)/60, 2)
    }
    np.save(f'results/{exp_path}/test_metrics_fold_{args.fold}.npy', metrics)
    
    time.sleep(1)
    
if __name__ == "__main__":
    main()
