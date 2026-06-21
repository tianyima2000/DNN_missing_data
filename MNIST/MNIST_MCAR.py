import numpy as np
import torch
import warnings
import pandas as pd
import os
import random
import pickle
import csv
from xgboost import XGBClassifier

import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.utils.prune as prune
from torchvision import datasets

from missforest.missforest import MissForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            restore_best_weights (bool): Whether to restore model weights from the best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        score = -val_loss  # We use negative because greater score = better performance
        
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = self._get_weights(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = self._get_weights(model)
            self.counter = 0
            
    def _get_weights(self, model):
        return {name: param.clone().detach() for name, param in model.state_dict().items()}
    
    def restore_weights(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)



def prune_and_reinit(model, amount, init_fn=nn.init.kaiming_uniform_):
    """
    Globally prune the model (magnitude pruning) except for layers under model.f2,
    keep the pruning masks, and reinitialise the surviving (unpruned) weights.

    Parameters
    ----------
    model : nn.Module
    amount : float
        Fraction of parameters to prune globally (e.g. 0.9 for 90%)
    init_fn : callable
        Weight initialisation for the surviving weights
    """
    # ---- 1. Build prune list (excluding f2) ----
    parameters_to_prune = []
    for name, module in model.named_modules():
        if name == "f2" or name.startswith("f2."):
            continue
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            parameters_to_prune.append((module, "weight"))

    # ---- 2. Global unstructured pruning ----
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # ---- 3. Reinitialise only the unpruned weights ----
    with torch.no_grad():
        for name, module in model.named_modules():
            if name == "f2" or name.startswith("f2."):
                continue

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, "weight_orig") and hasattr(module, "weight_mask"):
                    init_fn(module.weight_orig)   # reinit all
                    module.weight_orig.mul_(module.weight_mask)  # enforce sparsity

    return model
    


# def global_prune(model, amount):
#     # Collect all Linear layers and their weights
#     parameters_to_prune = [
#         (module, 'weight') 
#         for module in model.modules()
#         if isinstance(module, nn.Linear)
#     ]

#     # Perform global L1 unstructured pruning
#     prune.global_unstructured(
#         parameters_to_prune,
#         pruning_method=prune.L1Unstructured,
#         amount=amount,
#     )

#     # Process each pruned layer
#     for module, param_name in parameters_to_prune:
#         # Extract and store mask as buffer
#         mask = getattr(module, f"{param_name}_mask").clone()
#         module.register_buffer("pruning_mask", mask)
        
#         # Reinitialize unpruned weights while preserving zeros
#         with torch.no_grad():
#             # Get current weights (already pruned)
#             weight = getattr(module, param_name)
            
#             # Create new initialization
#             new_weights = torch.empty_like(weight)
#             nn.init.kaiming_uniform_(new_weights, mode='fan_in', nonlinearity='relu')
            
#             # Apply mask and update weights
#             weight.data.copy_(new_weights * mask)

#         # Remove PyTorch's pruning buffers
#         prune.remove(module, param_name)

#     # Register forward pre-hook to maintain pruning
#     def apply_mask(module, inputs):
#         if hasattr(module, "pruning_mask"):
#             mask = module.pruning_mask
#             with torch.no_grad():
#                 module.weight.data.mul_(mask)  # In-place multiplication

#     # Add hooks to all pruned modules
#     for module, _ in parameters_to_prune:
#         if hasattr(module, "pruning_mask"):
#             module.register_forward_pre_hook(apply_mask)

#     return model



"""
This is a function that trains a neural network on the training data and returns its testing and validation loss

Inputs:
    model: the neural network to train
    Z: covariates (tensor)
    Y: response (tensor)
    lr: learning rate for Adam optimiser
    prune_amount: the proportion of weights to be set to zero
    Omega: If Omega=None, then a standard neural network is used. 
           If Omega is the observation pattern (tensor), then PENN is used.
    epochs: maximum number of epochs to train
    weight_decay: weight decay for Adam optimiser
    prune_start: the epoch that pruning starts
    patience: patience for early stopping
    live_plot: if True, then the training loss and validation losses will be ploted live

Output: testing loss and validation loss
"""
def train_test_model(model, Z_train, Z_val, Z_test, Y_train, Y_val, Y_test, lr, prune_amount, Omega_train=None, Omega_val=None, Omega_test=None, epochs=200, weight_decay=0.001, prune_start=10, patience=10, live_plot=False):
    
    ##### if Omega is not None, then we use PENN
    if Omega_train is not None:

        ### set up for training
        train_data = TensorDataset(Z_train, Omega_train, Y_train)
        train_loader = DataLoader(dataset = train_data, batch_size=200, shuffle=True)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

        ### start training
        for epoch in range(epochs):

            if epoch == prune_start:
                model = prune_and_reinit(model, amount = prune_amount)
            for z_batch, omega_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(z_batch, omega_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
            
            if epoch >= 10:
                model.eval()
                early_stopping(loss_fn(model(Z_val, Omega_val), Y_val).item(), model)
                model.train()
                if early_stopping.early_stop:
                    break
        
        early_stopping.restore_weights(model)
        ### return testing loss and validation loss as a dictionary
        model.eval()
        correct=0
        with torch.no_grad():
            output = model(Z_test, Omega_test)
            pred = output.argmax(dim=1)
            correct += pred.eq(Y_test).sum().item()
        return {"test_mce": 1 - correct / Y_test.shape[0], "val_loss": loss_fn(model(Z_val, Omega_val), Y_val)}


    ##### if Omega is None
    if Omega_train is None:

        ### set up for training
        train_data = TensorDataset(Z_train, Y_train)
        train_loader = DataLoader(dataset = train_data, batch_size=200, shuffle=True)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

        ### start training
        for epoch in range(epochs):

            if epoch == prune_start:
                model = prune_and_reinit(model, amount = prune_amount)
            for z_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(z_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()

            if epoch >= 10:
                model.eval()
                early_stopping(loss_fn(model(Z_val), Y_val).item(), model)
                model.train()
                if early_stopping.early_stop:
                    break
        
        early_stopping.restore_weights(model)
        ### return testing loss and validation loss as a dictionary
        model.eval()
        correct=0
        with torch.no_grad():
            output = model(Z_test)
            pred = output.argmax(dim=1)
            correct += pred.eq(Y_test).sum().item()
        return {"test_mce": 1 - correct / Y_test.shape[0], "val_loss": loss_fn(model(Z_val), Y_val)}
    

def train_test_best_model(model_class, Z_train, Z_val, Z_test, Y_train, Y_val, Y_test, lr, prune_amount_vec, Omega_train=None, Omega_val=None, Omega_test=None, epochs=200, weight_decay=0.001, prune_start=10, patience=10, live_plot=False):
    N = len(prune_amount_vec)
    test_loss = np.zeros(N)
    val_loss = np.zeros(N)
    for i in range(N):
        model = model_class()
        model = model.to(device)
        output = train_test_model(model=model, Z_train=Z_train, Z_val=Z_val, Z_test=Z_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, lr=lr, 
                                  prune_amount=prune_amount_vec[i], Omega_train=Omega_train, Omega_val=Omega_val, Omega_test=Omega_test, 
                                  epochs=epochs, weight_decay=weight_decay, prune_start=prune_start, patience=patience, live_plot=live_plot)
        test_loss[i] = output["test_mce"]
        val_loss[i] = output["val_loss"].item()
        
    min_index = np.argmin(val_loss)
    return test_loss[min_index]



n = 70000
d = 28*28

def ceil(x):
    return int(np.ceil(x))
width_1 = ceil(np.sqrt(0.8*n/2))   # note that training sample size is 0.8n
width_2 = ceil(width_1/5)
embedding_dim = min(ceil(np.sqrt(d)), ceil(width_1/20))


### PENN with convolution
class PENN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Construct the neural network f1
        self.f1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, width_1),
            nn.ReLU()
        )

        # Construct the neural network f2, i.e. the embedding function
        self.f2 = nn.Sequential(
            nn.Linear(7*7, width_2),  
            nn.ReLU(),
            nn.Linear(width_2, width_2),  
            nn.ReLU(),
            nn.Linear(width_2, embedding_dim)
        )

        
        # Construct the neural network f3
        self.f3 = nn.Sequential(
            nn.Linear(width_1 + embedding_dim, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),
            nn.ReLU(),
            nn.Linear(width_1, 10)  
        )
    
    # Combine f1, f2 and f3 to construct the Pattern Embedding Neural Network (PENN)
    def forward(self, z, omega):
        omega = omega.view(-1, 7*7)

        # compute the output of f1 and f2
        f1_output = self.f1(z)
        f2_output = self.f2(omega)
        
        # Concatenate the output of f1 and f2
        combined_features = torch.cat((f1_output, f2_output), dim=1)
        
        # Apply the combined network
        final_output = self.f3(combined_features)
        
        return final_output
    

### Standard neural network
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),
            nn.ReLU(),
            nn.Linear(width_1, 10) 
        )

    def forward(self, x):
        final_output = self.f1(x)
        return final_output
    



train_data = datasets.MNIST('./data', train=True, download=True)
X_train = train_data.data.float() / 255
Y_train = train_data.targets.to(torch.long)

test_data = datasets.MNIST('./data', train=False, download=True)
X_test = test_data.data.float() / 255
Y_test = test_data.targets.to(torch.long)

X = torch.cat((X_train, X_test), dim=0).to(device)
Y = torch.cat((Y_train, Y_test), dim=0).to(device)

def one_run():
    Omega = torch.tensor(np.random.binomial(1, 0.5, (70000, 7, 7)), dtype=torch.float32).to(device)
    Omega_mask = torch.tensor(np.zeros((70000, 28, 28)), dtype=torch.float32).to(device)

    for j in range(7):
        for k in range(7):
            Omega_mask[:, j*4:(j+1)*4, k*4:(k+1)*4] = Omega[:, j, k].unsqueeze(-1).unsqueeze(-1)

    Z = X * Omega_mask
    Z_MI = Z + Z.sum(dim=0) / Omega_mask.sum(dim=0) * (1 - Omega_mask)
    Z[Omega_mask == 0] = torch.nan

    Z_cpu = Z.detach().cpu()
    Z_MI_cpu = Z_MI.detach().cpu()
    Omega_cpu = Omega.detach().cpu()
    Y_cpu = Y.detach().cpu()

    Z_train, Z_test, Z_MI_train, Z_MI_test, Omega_train, Omega_test, Y_train, Y_test = train_test_split(Z_cpu, Z_MI_cpu, Omega_cpu, Y_cpu, test_size=0.1, random_state=42, stratify=Y_cpu)

    Z_train, Z_val, Z_MI_train, Z_MI_val, Omega_train, Omega_val, Y_train, Y_val = train_test_split(Z_train, Z_MI_train, Omega_train, Y_train, test_size=0.11111, random_state=42, stratify=Y_train)

    Z_train, Z_val, Z_test = Z_train.to(device), Z_val.to(device), Z_test.to(device)
    Z_MI_train, Z_MI_val, Z_MI_test = Z_MI_train.to(device), Z_MI_val.to(device), Z_MI_test.to(device)
    Omega_train, Omega_val, Omega_test = Omega_train.to(device), Omega_val.to(device), Omega_test.to(device)
    Y_train, Y_val, Y_test = Y_train.to(device), Y_val.to(device), Y_test.to(device)

    Z_train= Z_train.unsqueeze(1)
    Z_val= Z_val.unsqueeze(1)
    Z_test= Z_test.unsqueeze(1)

    Z_MI_train = Z_MI_train.unsqueeze(1)
    Z_MI_val   = Z_MI_val.unsqueeze(1)
    Z_MI_test  = Z_MI_test.unsqueeze(1)


    prune_amount_vec = [0.9, 0.8, 0.6, 0.2]

    PENN_MI_loss = train_test_best_model(PENN, Z_train=Z_MI_train, Z_val=Z_MI_val, Z_test=Z_MI_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, Omega_train=Omega_train, Omega_val=Omega_val, Omega_test=Omega_test, lr=0.001, weight_decay=1e-3)
    NN_MI_loss = train_test_best_model(NN, Z_train=Z_MI_train, Z_val=Z_MI_val, Z_test=Z_MI_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, lr=0.001, weight_decay=1e-3)

                          
                                               
    # flatten
    Xtr = Z_train.squeeze(1).reshape(Z_train.shape[0], -1).detach().cpu().numpy()
    Xva = Z_val.squeeze(1).reshape(Z_val.shape[0], -1).detach().cpu().numpy()
    Xte = Z_test.squeeze(1).reshape(Z_test.shape[0], -1).detach().cpu().numpy()

    ytr = Y_train.detach().cpu().numpy().astype(np.int64).ravel()
    yva = Y_val.detach().cpu().numpy().astype(np.int64).ravel()
    yte = Y_test.detach().cpu().numpy().astype(np.int64).ravel()

    best_val_mce = float("inf")
    best_test_mce = None

    for max_depth in [3, 6, 9, 12]:
        clf = XGBClassifier(
            max_depth=max_depth,
            objective="multi:softmax",   # directly outputs class labels
            num_class=10,
            eval_metric="mlogloss",      # training metric (independent of selection)
            tree_method="hist",
            n_estimators=300,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
        )

        clf.fit(Xtr, ytr)

        # validation MCE
        val_pred = clf.predict(Xva)
        val_mce = np.mean(val_pred != yva)

        if val_mce < best_val_mce:
            best_val_mce = val_mce

            # test MCE at best validation depth
            test_pred = clf.predict(Xte)
            xgb_test_mce = np.mean(test_pred != yte)                                           
                                               
    return PENN_MI_loss, NN_MI_loss, xgb_test_mce



def one_run_rf():
    Omega = torch.tensor(np.random.binomial(1, 0.5, (70000, 7, 7)), dtype=torch.float32).to(device)
    Omega_mask = torch.tensor(np.zeros((70000, 28, 28)), dtype=torch.float32).to(device)

    for j in range(7):
        for k in range(7):
            Omega_mask[:, j*4:(j+1)*4, k*4:(k+1)*4] = Omega[:, j, k].unsqueeze(-1).unsqueeze(-1)

    Z = X * Omega_mask
    Z_MI = Z + Z.sum(dim=0) / Omega_mask.sum(dim=0) * (1 - Omega_mask)
    Z[Omega_mask == 0] = torch.nan

    Z_cpu = Z.detach().cpu()
    Z_MI_cpu = Z_MI.detach().cpu()
    Omega_cpu = Omega.detach().cpu()
    Y_cpu = Y.detach().cpu()

    Z_train, Z_test, Z_MI_train, Z_MI_test, Omega_train, Omega_test, Y_train, Y_test = train_test_split(Z_cpu, Z_MI_cpu, Omega_cpu, Y_cpu, test_size=0.1, random_state=42, stratify=Y_cpu)

    Z_train, Z_val, Z_MI_train, Z_MI_val, Omega_train, Omega_val, Y_train, Y_val = train_test_split(Z_train, Z_MI_train, Omega_train, Y_train, test_size=0.11111, random_state=42, stratify=Y_train)

    Z_train, Z_val, Z_test = Z_train.to(device), Z_val.to(device), Z_test.to(device)
    Z_MI_train, Z_MI_val, Z_MI_test = Z_MI_train.to(device), Z_MI_val.to(device), Z_MI_test.to(device)
    Omega_train, Omega_val, Omega_test = Omega_train.to(device), Omega_val.to(device), Omega_test.to(device)
    Y_train, Y_val, Y_test = Y_train.to(device), Y_val.to(device), Y_test.to(device)

    Z_train= Z_train.unsqueeze(1)
    Z_val= Z_val.unsqueeze(1)
    Z_test= Z_test.unsqueeze(1)

    Z_MI_train = Z_MI_train.unsqueeze(1)
    Z_MI_val   = Z_MI_val.unsqueeze(1)
    Z_MI_test  = Z_MI_test.unsqueeze(1)

    # flatten
    Xtr = Z_train.squeeze(1).reshape(Z_train.shape[0], -1).detach().cpu().numpy()
    Xva = Z_val.squeeze(1).reshape(Z_val.shape[0], -1).detach().cpu().numpy()
    Xte = Z_test.squeeze(1).reshape(Z_test.shape[0], -1).detach().cpu().numpy()

    ytr = Y_train.detach().cpu().numpy().astype(np.int64).ravel()
    yva = Y_val.detach().cpu().numpy().astype(np.int64).ravel()
    yte = Y_test.detach().cpu().numpy().astype(np.int64).ravel()

    best_val_mce = float("inf")
    best_test_mce = None

    d= 28*28
    for max_features in [ceil(0.1*d), ceil(0.2*d), ceil(0.4*d), ceil(0.8*d)]:
        clf = RandomForestClassifier(max_features = max_features)

        clf.fit(Xtr, ytr)

        # validation MCE
        val_pred = clf.predict(Xva)
        val_mce = np.mean(val_pred != yva)

        if val_mce < best_val_mce:
            best_val_mce = val_mce

            # test MCE at best validation depth
            test_pred = clf.predict(Xte)
            rf_test_mce = np.mean(test_pred != yte)                                           
                                               
    return rf_test_mce
