import numpy as np
import torch
import warnings
import pandas as pd
import os
import random
import pickle
import csv
from xgboost import XGBRegressor

import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.utils.prune as prune

from missforest.missforest import MissForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from scipy.special import expit

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
        loss_fn = nn.MSELoss()
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
                early_stopping(loss_fn(model(Z_val, Omega_val), Y_val), model)
                if early_stopping.early_stop:
                    break
        
        early_stopping.restore_weights(model)
        ### return testing loss and validation loss as a dictionary
        model.eval()
        return {"test_loss": loss_fn(model(Z_test, Omega_test), Y_test), "val_loss": loss_fn(model(Z_val, Omega_val), Y_val)}


    ##### if Omega is None
    if Omega_train is None:

        ### set up for training
        train_data = TensorDataset(Z_train, Y_train)
        train_loader = DataLoader(dataset = train_data, batch_size=200, shuffle=True)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
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
                early_stopping(loss_fn(model(Z_val), Y_val), model)
                if early_stopping.early_stop:
                    break
        
        early_stopping.restore_weights(model)
        ### return testing loss and validation loss as a dictionary
        model.eval()
        return {"test_loss": loss_fn(model(Z_test), Y_test), "val_loss": loss_fn(model(Z_val), Y_val)}
    

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
        test_loss[i] = output["test_loss"].item()
        val_loss[i] = output["val_loss"].item()
        
    min_index = np.argmin(val_loss)
    return test_loss[min_index]



df = pd.read_csv("/home/tm681/DNN_missing_data/relative_location_CT/slice_localization_data.csv")
X = df.drop(columns=['patientId', 'reference'])
scaler = StandardScaler()
X = scaler.fit_transform(X.to_numpy())
Y  = df['reference']

n = X.shape[0]
d = X.shape[1]

def ceil(x):
    return int(np.ceil(x))
width_1 = ceil(np.sqrt(0.8*n/2))   # note that training sample size is 0.8n
width_2 = ceil(width_1/5)
embedding_dim = min(ceil(np.sqrt(d)), ceil(width_1/20))

### Pattern Embedding Neural Network (PENN)
class PENN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Construct the neural network f1
        self.f1 = nn.Sequential(
            nn.Linear(d, width_1),  
            nn.ReLU(),
            nn.Linear(width_1, width_1),  
            nn.ReLU(),
            nn.Linear(width_1, width_1),  
            nn.ReLU(),
        )

        # Construct the neural network f2, i.e. the embedding function
        self.f2 = nn.Sequential(
            nn.Linear(d, width_2),  
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
            nn.Linear(width_1, 1)  
        )
    
    # Combine f1, f2 and f3 to construct the Pattern Embedding Neural Network (PENN)
    def forward(self, z, omega):
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
        self.nn = nn.Sequential(
            nn.Linear(d, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),  
            nn.ReLU(),
            nn.Linear(width_1,1)
        )
    
    def forward(self, x):
        final_output = self.nn(x)
        return final_output
    


def one_run():
    probs = expit(-2*X + 1 + 4*X.mean(axis=1, keepdims=True))
    Omega = np.random.binomial(1, probs)
    Z = pd.DataFrame(X)
    Z = Z.mask(Omega==0)

    Z_MI = Z.copy(deep=True)
    Z_MI = Z_MI.fillna(Z.mean())
    scaler = StandardScaler()
    Z_MI = scaler.fit_transform(Z_MI.to_numpy())

    # Z_MF = Z.copy(deep=True)
    # warnings.filterwarnings("ignore")
    # rgr = RandomForestRegressor(n_estimators=10, n_jobs=-1)
    # mf_imputer = MissForest(rgr, max_iter=10,verbose=False)
    # Z_MF = mf_imputer.fit_transform(Z_MF)
    # scaler = StandardScaler()
    # Z_MF = scaler.fit_transform(Z_MF)

    Z_II = Z.copy(deep=True)
    II_imputer = IterativeImputer(max_iter=10, n_nearest_features=20)
    Z_II = II_imputer.fit_transform(Z_II.to_numpy())
    scaler = StandardScaler()
    Z_II = scaler.fit_transform(Z_II)

    Z_train, Z_test, Z_MI_train, Z_MI_test, Z_II_train, Z_II_test, Omega_train, Omega_test, Y_train, Y_test = train_test_split(Z, Z_MI, Z_II, Omega, Y, test_size=0.1)
    Z_train, Z_val, Z_MI_train, Z_MI_val, Z_II_train, Z_II_val, Omega_train, Omega_val, Y_train, Y_val = train_test_split(Z_train, Z_MI_train, Z_II_train, Omega_train, Y_train, test_size=0.11111)
    
    Z_MI_train = torch.tensor(Z_MI_train, dtype=torch.float32)
    Z_MI_val = torch.tensor(Z_MI_val, dtype=torch.float32)
    Z_MI_test = torch.tensor(Z_MI_test, dtype=torch.float32)
    
    # Z_MF_train = torch.tensor(Z_MF_train, dtype=torch.float32)
    # Z_MF_val = torch.tensor(Z_MF_val, dtype=torch.float32)
    # Z_MF_test = torch.tensor(Z_MF_test, dtype=torch.float32)
    
    Z_II_train = torch.tensor(Z_II_train, dtype=torch.float32)
    Z_II_val = torch.tensor(Z_II_val, dtype=torch.float32)
    Z_II_test = torch.tensor(Z_II_test, dtype=torch.float32)
    
    Omega_train = torch.tensor(Omega_train, dtype=torch.float32)
    Omega_val = torch.tensor(Omega_val, dtype=torch.float32)
    Omega_test = torch.tensor(Omega_test, dtype=torch.float32)
    Y_train = torch.tensor(Y_train.to_numpy().reshape(-1,1), dtype=torch.float32)
    Y_val = torch.tensor(Y_val.to_numpy().reshape(-1,1), dtype=torch.float32)
    Y_test = torch.tensor(Y_test.to_numpy().reshape(-1,1), dtype=torch.float32)

    prune_amount_vec = [0.9, 0.8, 0.6, 0.2]

    PENN_MI_loss = train_test_best_model(PENN, Z_train=Z_MI_train, Z_val=Z_MI_val, Z_test=Z_MI_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, Omega_train=Omega_train, Omega_val=Omega_val, Omega_test=Omega_test, lr=0.001, weight_decay=1e-3, patience=10)
    NN_MI_loss = train_test_best_model(NN, Z_train=Z_MI_train, Z_val=Z_MI_val, Z_test=Z_MI_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, lr=0.001, weight_decay=1e-3, patience=10)
    
    # PENN_MF_loss = train_test_best_model(PENN, Z_train=Z_MF_train, Z_val=Z_MF_val, Z_test=Z_MF_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
    #                                            prune_amount_vec=prune_amount_vec, Omega_train=Omega_train, Omega_val=Omega_val, Omega_test=Omega_test, lr=0.001, weight_decay=1e-3, patience=10)
    # NN_MF_loss = train_test_best_model(NN, Z_train=Z_MF_train, Z_val=Z_MF_val, Z_test=Z_MF_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
    #                                            prune_amount_vec=prune_amount_vec, lr=0.001, weight_decay=1e-3, patience=10)
                                               
    PENN_II_loss = train_test_best_model(PENN, Z_train=Z_II_train, Z_val=Z_II_val, Z_test=Z_II_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, Omega_train=Omega_train, Omega_val=Omega_val, Omega_test=Omega_test, lr=0.001, weight_decay=1e-3, patience=10)
    NN_II_loss = train_test_best_model(NN, Z_train=Z_II_train, Z_val=Z_II_val, Z_test=Z_II_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, lr=0.001, weight_decay=1e-3, patience=10)
                          
                                               
    best_xgb_val_mse = float("inf")
    for max_depth in [3, 6, 9, 12]:
        model = XGBRegressor(max_depth=max_depth, enable_categorical=True)
        model.fit(Z_train, Y_train.detach().cpu().numpy().ravel())
        xgb_val_mse = np.mean((Y_val.detach().cpu().numpy().ravel() - model.predict(Z_val))**2)
        if xgb_val_mse < best_xgb_val_mse:
            best_xgb_val_mse = xgb_val_mse
            xgb_test_mse = np.mean((Y_test.detach().cpu().numpy().ravel() - model.predict(Z_test))**2)                                           
                                               
    return PENN_MI_loss / Y_test.var(unbiased=False).item(), NN_MI_loss / Y_test.var(unbiased=False).item(), PENN_II_loss / Y_test.var(unbiased=False).item(), NN_II_loss / Y_test.var(unbiased=False).item(), xgb_test_mse / Y_test.var(unbiased=False).item()



def one_run_rf():
    probs = expit(-2*X + 1 + 4*X.mean(axis=1, keepdims=True))
    Omega = np.random.binomial(1, probs)
    Z = pd.DataFrame(X)
    Z = Z.mask(Omega==0)

    Z_train, Z_test, Y_train, Y_test = train_test_split(Z, Y, test_size=0.1)
    Z_train, Z_val, Y_train, Y_val = train_test_split(Z_train, Y_train, test_size=0.11111)
                                               
    best_rf_val_mse = float("inf")
    for max_features in [ceil(0.1*d), ceil(0.2*d), ceil(0.4*d), ceil(0.8*d)]:
        model = RandomForestRegressor(max_features=max_features)
        model.fit(Z_train, Y_train)
        rf_val_mse = np.mean((Y_val - model.predict(Z_val))**2)
        if rf_val_mse < best_rf_val_mse:
            best_rf_val_mse = rf_val_mse
            rf_test_mse = np.mean((Y_test - model.predict(Z_test))**2)                                           
                                               
    return rf_test_mse / Y_test.var()
