"""
This code creates a Flower client that can be used to train a model locally and share the updated 
model with the server. When it is started, it connects to the Flower server and waits for instructions.
If the server sends a model, the client trains the model locally and sends back the updated model.
If abilitated, at the end of the training the client evaluates the last model, and plots the 
metrics during the training.

This is code is set to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
have this code running.
"""

import argparse
import numpy as np
import copy
from collections import OrderedDict
import json

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import flwr as fl

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import public.config as cfg
import public.utils as utils
import public.models as models


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
        model,
        client_id, 
        device
        ):
        self.model = model
        self.global_model = copy.deepcopy(model)
        self.global_model_loaded = False
        self.client_id = client_id
        self.device = device
        self.drifting_log = []

        # plot
        self.metrics = {
            "rounds": [],
            "loss": [],
            "accuracy": []
        }

        if cfg.training_drifting:
            drifting_log = np.load(f'../data/cur_datasets/drifting_log.npy', allow_pickle=True).item()
            self.drifting_log = drifting_log[self.client_id]

    def load_current_data(self,
                          cur_round,
                          train=True,
                          descriptor_extraction=False) -> DataLoader:
        # load raw data
        if not cfg.training_drifting:
            cur_data = np.load(f'../data/cur_datasets/client_{self.client_id}.npy', allow_pickle=True).item()
        else:
            load_index = max([index for index in self.drifting_log if index <= cur_round], default=0)
            cur_data = np.load(f'../data/cur_datasets/client_{self.client_id}_round_{load_index}.npy', allow_pickle=True).item()
        
        cur_features = torch.tensor(cur_data['train_features'], dtype=torch.float32) if not cfg.training_drifting else torch.tensor(cur_data['features'], dtype=torch.float32)
        cur_labels = torch.tensor(cur_data['train_labels'], dtype=torch.int64) if not cfg.training_drifting else torch.tensor(cur_data['labels'], dtype=torch.int64)

        if not cfg.dataset_name == "CheXpert":
            cur_features = cur_features.unsqueeze(1) if utils.get_in_channels() == 1 else cur_features

        # Split the data into training and testing subsets
        train_features, val_features, train_labels, val_labels = train_test_split(
            cur_features, cur_labels, test_size=cfg.client_eval_ratio, random_state=cfg.random_seed
        )
        
        # reduce the number of samples 
        if cfg.n_samples_clients > 0:
            train_features = train_features[:cfg.n_samples_clients]
            train_labels = train_labels[:cfg.n_samples_clients]

        if train:
            train_dataset = models.CombinedDataset(train_features, train_labels, transform=None)
            return DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        else:
            val_dataset = models.CombinedDataset(val_features, val_labels, transform=None)
            # randomly sample the data for descriptor extraction
            if descriptor_extraction:
                if cfg.n_stochastic_sampling > 0:
                    val_dataset_list = []
                    for i in range(cfg.n_stochastic_sampling):
                        val_subset, _ = train_test_split(val_dataset, test_size=0.5, random_state= cfg.random_seed + cur_round**4 + i)
                        val_dataset_list.append(val_subset)
                    val_dataset = torch.utils.data.ConcatDataset(val_dataset_list)
            return DataLoader(val_dataset, batch_size=cfg.test_batch_size, shuffle=False)

    # override
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # override
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # override
    def fit(self, parameters, config):
        cur_round = config["current_round"]
        
        if config["extract_descriptors"] and (config["fedavg"] == False):
            samples_val_loader = self.load_current_data(cur_round, train=False, descriptor_extraction=True)

            # set the global model on the client to extract descriptors
            if self.global_model_loaded == False:
                self.global_model_loaded = True
                self.global_model.load_state_dict(torch.load(f"checkpoints/{cfg.default_path}/{cfg.non_iid_type}_global_model.pth", weights_only=False))
            
            # Extract descriptors

            descriptors = models.ModelEvaluator(test_loader=samples_val_loader, device=self.device).extract_descriptors(model=self.global_model, \
                                                            client_id=self.client_id, max_latent_space=config["max_latent_space"]) 

            # # save client descriptors and samples_val_loader
            # d = json.loads(descriptors["latent_space_mean"]) + json.loads(descriptors["latent_space_std"]) + json.loads(descriptors["latent_space_mean_by_label"]) + json.loads(descriptors["latent_space_std_by_label"])
            # # convert to numpy
            # d = np.array(d)
            # foldd = "mnist_epsilon_umap/temp_Py_scaling1_ep5/"
            # os.makedirs(foldd, exist_ok=True)
            # np.save(foldd + f"descriptor_client_{self.client_id}_round_{cur_round}.npy", d)
            # # np.save(f"temp/data_client_{self.client_id}_round_{cur_round}.npy", samples_val_loader.dataset[:])
            # features  = torch.stack([feat for feat, _ in samples_val_loader.dataset]).numpy()
            # labels    = torch.tensor([label for _, label in samples_val_loader.dataset]).numpy()
            # np.save(foldd + f"features_client_{self.client_id}_round_{cur_round}.npy", features)
            # np.save(foldd + f"labels_client_{self.client_id}_round_{cur_round}.npy", labels)
                    
            return self.last_trained_parameters, len(samples_val_loader.dataset), descriptors

        else: 
            cur_train_loader = self.load_current_data(cur_round, train=True)

            # set updated/aggregated parameters
            self.set_parameters(parameters)

            # Train the model   
            for epoch in range(config["local_epochs"]):
                models.simple_train(model=self.model,
                                    device=self.device,
                                    train_loader=cur_train_loader, 
                                    optimizer=torch.optim.SGD(self.model.parameters(), lr=cfg.lr, momentum=cfg.momentum),
                                    epoch=epoch,
                                    client_id=self.client_id)
            
            # Evaluate the model
            # ...
            
            # save the last trained parameters
            self.last_trained_parameters = self.get_parameters(config)
            # i return self.last_trained_parameters to avoid the error but not necessary
            return self.last_trained_parameters, len(cur_train_loader.dataset), {}
        
    def evaluate(self, parameters, config):
        cur_round = config["current_round"]
        cur_val_loader = self.load_current_data(cur_round, train=False)
        
        # set the last trained parameters
        self.set_parameters(self.last_trained_parameters)

        # Evaluate the model
        loss_trad, accuracy_trad, f1_score_trad, new_max_latent_space = \
            models.ModelEvaluator(test_loader=cur_val_loader, device=self.device).evaluate(self.model)    

        # quick check results and save for plot
        print(f"Client {self.client_id} - Round {cur_round} - Loss: {loss_trad:.4f}, Accuracy: {accuracy_trad:.4f}")
        self.metrics["rounds"].append(cur_round)
        self.metrics["loss"].append(loss_trad)
        self.metrics["accuracy"].append(accuracy_trad)
        np.save(f"results/{cfg.default_path}/client_{self.client_id}_metrics.npy", self.metrics)

        return float(loss_trad), len(cur_val_loader.dataset), {
            "accuracy": float(accuracy_trad),
            "f1_score": float(f1_score_trad),
            "max_latent_space": float(new_max_latent_space),
            "cid": int(self.client_id),
            "round": int(cur_round)
        }


# main
def main() -> None:
    # Get client id
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--id", type=int, choices=range(0, cfg.n_clients), required=True,
        help="Specifies the artificial data partition",
    )
    parser.add_argument("--fold", type=int, required=False, default=0,
        help="Specifies the fold number of the cross-validation",
    )
    args = parser.parse_args()

    # Load device, model and data
    utils.set_seed(cfg.random_seed + args.fold)
    device = utils.check_gpu(client_id=args.id)
    in_channels = utils.get_in_channels()
    model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                          input_size=cfg.input_size).to(device)

    # Start Flower client
    client = FlowerClient(model=model,
                          client_id=args.id,
                          device=device
                          ).to_client()
    
    fl.client.start_client(server_address=f"{cfg.ip}:{cfg.port}", client=client) # local host

if __name__ == "__main__":
    main()
