import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import pandas as pd
import os
import random
import pickle

from IPython.display import clear_output

import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.utils.prune as prune

from missforest.missforest import MissForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



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

    


def global_prune(model, amount):
    # Collect all Linear layers and their weights
    parameters_to_prune = [
        (module, 'weight') 
        for module in model.modules()
        if isinstance(module, nn.Linear)
    ]

    # Perform global L1 unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Process each pruned layer
    for module, param_name in parameters_to_prune:
        # Extract and store mask as buffer
        mask = getattr(module, f"{param_name}_mask").clone()
        module.register_buffer("pruning_mask", mask)
        
        # Reinitialize unpruned weights while preserving zeros
        with torch.no_grad():
            # Get current weights (already pruned)
            weight = getattr(module, param_name)
            
            # Create new initialization
            new_weights = torch.empty_like(weight)
            nn.init.kaiming_uniform_(new_weights, mode='fan_in', nonlinearity='relu')
            
            # Apply mask and update weights
            weight.data.copy_(new_weights * mask)

        # Remove PyTorch's pruning buffers
        prune.remove(module, param_name)

    # Register forward pre-hook to maintain pruning
    def apply_mask(module, inputs):
        if hasattr(module, "pruning_mask"):
            mask = module.pruning_mask
            with torch.no_grad():
                module.weight.data.mul_(mask)  # In-place multiplication

    # Add hooks to all pruned modules
    for module, _ in parameters_to_prune:
        if hasattr(module, "pruning_mask"):
            module.register_forward_pre_hook(apply_mask)

    return model



class QuantileLoss2D(nn.Module):
    def __init__(self, quantiles):
        """
        Args:
            quantiles: A pair [q1, q2] specifying the quantile levels for each output coordinate.
        """
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Predicted values, shape [batch_size, 2]
            y_true: True y values, shape [batch_size, 1]
        """
        error = y_true - y_pred  # Shape: (batch_size, 2)
        loss = torch.maximum(self.quantiles * error, (self.quantiles - 1) * error)
        return loss.mean()  # Averaging over all samples and output dimensions
    


def CQR(quantiles_pred, y_true, alpha):
    with torch.no_grad():
        quantiles_pred = quantiles_pred.numpy()
        y_true = y_true.numpy().flatten()

    E = np.zeros(len(y_true))
    for i in range(len(y_true)):
        E[i] = np.max([quantiles_pred[i,0] - y_true[i], y_true[i] - quantiles_pred[i,1]])
    E = np.sort(E)
    index = int(np.ceil((1-alpha) * (1+1/len(E)) * len(E)))
    return E[index]




"""
This is a function that trains a neural network on the training data and returns its validation loss and prediction intervals on test set

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
def train_test_model(model, Z_train, Z_val, Z_cal, Z_test, Y_train, Y_val, Y_cal, Y_test, lr, prune_amount, Omega_train=None, Omega_val=None, Omega_cal=None, Omega_test=None, epochs=200, weight_decay=0.001, prune_start=10, patience=10, live_plot=False):

    train_losses = []
    val_losses = []
    # Set up interactive plotting
    if live_plot:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        line1, = ax.plot(train_losses, 'b-', label='Training Loss')
        line2, = ax.plot(val_losses, 'r-', label='Validation Loss')
        plt.legend()
        plt.show()
    
    ##### if Omega is not None, then we use PENN
    if Omega_train is not None:

        ### set up for training
        train_data = TensorDataset(Z_train, Omega_train, Y_train)
        train_loader = DataLoader(dataset = train_data, batch_size=200, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = QuantileLoss2D([0.05, 0.95])
        early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

        ### start training
        for epoch in range(epochs):

            model.train()
            if epoch == prune_start:
                model = global_prune(model, amount = prune_amount)
            for z_batch, omega_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(z_batch, omega_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                
            
            model.eval()
            with torch.no_grad():
                train_losses.append(loss_fn(model(Z_train, Omega_train), Y_train))
                val_losses.append(loss_fn(model(Z_val, Omega_val), Y_val))

            # Update plot
            if live_plot:
                ax.set_title(f'epoch {epoch+1}/{epochs}')
                if epoch >= 4:
                    line1.set_ydata(train_losses[4:])
                    line1.set_xdata(range(5,epoch+2))
                    line2.set_ydata(val_losses[4:])
                    line2.set_xdata(range(5,epoch+2))
                    ax.relim()  # Recalculate limits
                    ax.autoscale_view(True, True, True)  # Autoscale
                
                # create plot
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1) 

            if epoch >= 10:
                early_stopping(loss_fn(model(Z_val, Omega_val), Y_val), model)
                if early_stopping.early_stop:
                    break
             

        # close the plot
        if live_plot:
            plt.ioff()
            plt.close(fig)   
        
        early_stopping.restore_weights(model)
        ### return testing loss and validation loss as a dictionary
        model.eval()
        with torch.no_grad():
            pred_intervals = model(Z_test, Omega_test).numpy()
            pred_intervals[:,0] = pred_intervals[:,0] - CQR(model(Z_cal, Omega_cal), Y_cal, 0.1)
            pred_intervals[:,1] = pred_intervals[:,1] + CQR(model(Z_cal, Omega_cal), Y_cal, 0.1)
        return {"val_loss": loss_fn(model(Z_val, Omega_val), Y_val), "pred_intervals": pred_intervals}


    ##### if Omega is None
    if Omega_train is None:

        ### set up for training
        train_data = TensorDataset(Z_train, Y_train)
        train_loader = DataLoader(dataset = train_data, batch_size=200, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = QuantileLoss2D([0.05, 0.95])
        early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

        ### start training
        for epoch in range(epochs):

            model.train()
            if epoch == prune_start:
                model = global_prune(model, amount = prune_amount)
            for z_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(z_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                
            
            model.eval()
            with torch.no_grad():
                train_losses.append(loss_fn(model(Z_train), Y_train))
                val_losses.append(loss_fn(model(Z_val), Y_val))

            # Update plot
            if live_plot:
                ax.set_title(f'epoch {epoch+1}/{epochs}')
                if epoch >= 4:
                    line1.set_ydata(train_losses[4:])
                    line1.set_xdata(range(5,epoch+2))
                    line2.set_ydata(val_losses[4:])
                    line2.set_xdata(range(5,epoch+2))
                    ax.relim()  # Recalculate limits
                    ax.autoscale_view(True, True, True)  # Autoscale
                
                # create plot
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1) 

            if epoch >= 10:
                early_stopping(loss_fn(model(Z_val), Y_val), model)
                if early_stopping.early_stop:
                    break
    

        # close the plot
        if live_plot:
            plt.ioff()
            plt.close(fig)   
        
        early_stopping.restore_weights(model)
        ### return testing loss and validation loss as a dictionary
        model.eval()
        with torch.no_grad():
            pred_intervals = model(Z_test).numpy()
            pred_intervals[:,0] = pred_intervals[:,0] - CQR(model(Z_cal), Y_cal, 0.1)
            pred_intervals[:,1] = pred_intervals[:,1] + CQR(model(Z_cal), Y_cal, 0.1)
        return {"val_loss": loss_fn(model(Z_val), Y_val), "pred_intervals": pred_intervals}
    

def train_test_best_model(model_class, Z_train, Z_val, Z_cal, Z_test, Y_train, Y_val, Y_cal, Y_test, lr, prune_amount_vec, Omega_train=None, Omega_val=None, Omega_cal=None, Omega_test=None, epochs=200, weight_decay=0.001, prune_start=10, patience=10, live_plot=False):
    N = len(prune_amount_vec)
    val_loss = np.zeros(N)
    pred_intervals = {}
    for i in range(N):
        model = model_class()
        output = train_test_model(model=model, Z_train=Z_train, Z_val=Z_val, Z_cal=Z_cal, Z_test=Z_test, Y_train=Y_train, Y_val=Y_val, Y_cal=Y_cal, Y_test=Y_test, lr=lr, 
                                  prune_amount=prune_amount_vec[i], Omega_train=Omega_train, Omega_val=Omega_val, Omega_cal=Omega_cal, Omega_test=Omega_test, 
                                  epochs=epochs, weight_decay=weight_decay, prune_start=prune_start, patience=patience, live_plot=live_plot)
        val_loss[i] = output["val_loss"]
        pred_intervals[i] = output["pred_intervals"]
        
    min_index = np.argmin(val_loss)
    return pred_intervals[min_index]



# Pattern Embedding Neural Networks (PENN)
class PENN(nn.Module):
    def __init__(self):
        super().__init__()
        inputdim = 304
        width1 = width3 = 100
        width2 = 30
        embedding_dim = 3
        
        # Construct the neural network f1
        self.f1 = nn.Sequential(
            nn.Linear(inputdim, width1),  
            nn.ReLU(),
            nn.Linear(width1, width1), 
            nn.ReLU(),
            nn.Linear(width1, width1), 
            nn.ReLU()
        )

        # Construct the neural network f2, i.e. the embedding function
        self.f2 = nn.Sequential(
            nn.Linear(inputdim, width2),  
            nn.ReLU(),
            nn.Linear(width2, width2),  
            nn.ReLU(),
            nn.Linear(width2, embedding_dim)
        )

        
        # Construct the neural network f3
        self.f3 = nn.Sequential(
            nn.Linear(width1 + embedding_dim, width3),
            nn.ReLU(),
            nn.Linear(width3, width3),
            nn.ReLU(),
            nn.Linear(width3, width3),
            nn.ReLU(),
            nn.Linear(width3, 2)  
        )
    
    # Combine f1, f2 and f3 to construct the Pattern Embedding Neural Network
    def forward(self, z, omega):
        # Compute the output of f1 and f2
        f1_output = self.f1(z)
        f2_output = self.f2(omega)
        
        # Concatenate the output of f1 and f2
        combined_features = torch.cat((f1_output, f2_output), dim=1)
        
        # Apply the combined network
        final_output = self.f3(combined_features)
        
        return final_output
    

# Standard neural network
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        inputdim = 304
        width = 100

        self.nn = nn.Sequential(
            nn.Linear(inputdim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width,2)
        )
    
    def forward(self, x):
        final_output = self.nn(x)
        return final_output
    

with open("C:/Users/marti/Desktop/DNN_missing_data/credit-score-prediction/credit_score_data.pkl", 'rb') as f:
    data = pickle.load(f)

Z_MI = data['Z_MI']
Z_MF = data['Z_MF']
Z_II = data['Z_II']
Omega = data['Omega']
Y = data['Y']


total_iterations = 10

PENN_MI_width = np.zeros(total_iterations)
NN_MI_width = np.zeros(total_iterations)
PENN_MF_width = np.zeros(total_iterations)
NN_MF_width = np.zeros(total_iterations)
PENN_II_width = np.zeros(total_iterations)
NN_II_width = np.zeros(total_iterations)

PENN_MI_coverage = np.zeros(total_iterations)
NN_MI_coverage = np.zeros(total_iterations)
PENN_MF_coverage = np.zeros(total_iterations)
NN_MF_coverage = np.zeros(total_iterations)
PENN_II_coverage = np.zeros(total_iterations)
NN_II_coverage = np.zeros(total_iterations)

for iter in tqdm(range(total_iterations), bar_format='[{elapsed}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}'):
    Z_MI_train, Z_MI_test, Z_MF_train, Z_MF_test, Z_II_train, Z_II_test, Omega_train, Omega_test, Y_train, Y_test = train_test_split(Z_MI, Z_MF, Z_II, Omega, Y, test_size=0.1)
    Z_MI_train, Z_MI_val, Z_MF_train, Z_MF_val, Z_II_train, Z_II_val, Omega_train, Omega_val, Y_train, Y_val = train_test_split(Z_MI_train, Z_MF_train, Z_II_train, Omega_train, Y_train, test_size=0.11111)
    Z_MI_train, Z_MI_cal, Z_MF_train, Z_MF_cal, Z_II_train, Z_II_cal, Omega_train, Omega_cal, Y_train, Y_cal = train_test_split(Z_MI_train, Z_MF_train, Z_II_train, Omega_train, Y_train, test_size=0.125)

    Z_MI_train = torch.tensor(Z_MI_train.to_numpy(), dtype=torch.float32)
    Z_MI_val = torch.tensor(Z_MI_val.to_numpy(), dtype=torch.float32)
    Z_MI_cal = torch.tensor(Z_MI_cal.to_numpy(), dtype=torch.float32)
    Z_MI_test = torch.tensor(Z_MI_test.to_numpy(), dtype=torch.float32)

    Z_MF_train = torch.tensor(Z_MF_train.to_numpy(), dtype=torch.float32)
    Z_MF_val = torch.tensor(Z_MF_val.to_numpy(), dtype=torch.float32)
    Z_MF_cal = torch.tensor(Z_MF_cal.to_numpy(), dtype=torch.float32)
    Z_MF_test = torch.tensor(Z_MF_test.to_numpy(), dtype=torch.float32)

    Z_II_train = torch.tensor(Z_II_train.to_numpy(), dtype=torch.float32)
    Z_II_val = torch.tensor(Z_II_val.to_numpy(), dtype=torch.float32)
    Z_II_cal = torch.tensor(Z_II_cal.to_numpy(), dtype=torch.float32)
    Z_II_test = torch.tensor(Z_II_test.to_numpy(), dtype=torch.float32)

    Omega_train = torch.tensor(Omega_train.to_numpy(), dtype=torch.float32)
    Omega_val = torch.tensor(Omega_val.to_numpy(), dtype=torch.float32)
    Omega_cal = torch.tensor(Omega_cal.to_numpy(), dtype=torch.float32)
    Omega_test = torch.tensor(Omega_test.to_numpy(), dtype=torch.float32)

    Y_train = torch.tensor(Y_train.to_numpy().reshape(-1,1), dtype=torch.float32)
    Y_val = torch.tensor(Y_val.to_numpy().reshape(-1,1), dtype=torch.float32)
    Y_cal = torch.tensor(Y_cal.to_numpy().reshape(-1,1), dtype=torch.float32)
    Y_test = torch.tensor(Y_test.to_numpy().reshape(-1,1), dtype=torch.float32)

    prune_amount_vec = [0.9, 0.8, 0.6, 0.2]

    PENN_MI = train_test_best_model(PENN, Z_train=Z_MI_train, Z_val=Z_MI_val, Z_cal=Z_MI_cal, Z_test=Z_MI_test, Y_train=Y_train, Y_val=Y_val, Y_cal=Y_cal, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, Omega_train=Omega_train, Omega_val=Omega_val, Omega_cal=Omega_cal, Omega_test=Omega_test, lr=0.001)
    PENN_MI_width[iter] = np.mean(PENN_MI[:,1] - PENN_MI[:,0])
    PENN_MI_coverage[iter] = np.mean([Y_test[i].item() >= PENN_MI[i, 0] and Y_test[i].item() <= PENN_MI[i, 1] for i in range(Y_test.shape[0])])
    

    NN_MI = train_test_best_model(NN, Z_train=Z_MI_train, Z_val=Z_MI_val, Z_cal=Z_MI_cal, Z_test=Z_MI_test, Y_train=Y_train, Y_val=Y_val, Y_cal=Y_cal, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, lr=0.001)
    NN_MI_width[iter] = np.mean(NN_MI[:,1] - NN_MI[:,0])
    NN_MI_coverage[iter] = np.mean([Y_test[i].item() >= NN_MI[i, 0] and Y_test[i].item() <= NN_MI[i, 1] for i in range(Y_test.shape[0])])
    
    PENN_MF = train_test_best_model(PENN, Z_train=Z_MF_train, Z_val=Z_MF_val, Z_cal=Z_MF_cal, Z_test=Z_MF_test, Y_train=Y_train, Y_val=Y_val, Y_cal=Y_cal, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, Omega_train=Omega_train, Omega_val=Omega_val, Omega_cal=Omega_cal, Omega_test=Omega_test, lr=0.001)
    PENN_MF_width[iter] = np.mean(PENN_MF[:,1] - PENN_MF[:,0])
    PENN_MF_coverage[iter] = np.mean([Y_test[i].item() >= PENN_MF[i, 0] and Y_test[i].item() <= PENN_MF[i, 1] for i in range(Y_test.shape[0])])
    

    NN_MF = train_test_best_model(NN, Z_train=Z_MF_train, Z_val=Z_MF_val, Z_cal=Z_MF_cal, Z_test=Z_MF_test, Y_train=Y_train, Y_val=Y_val, Y_cal=Y_cal, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, lr=0.001)
    NN_MF_width[iter] = np.mean(NN_MF[:,1] - NN_MF[:,0])
    NN_MF_coverage[iter] = np.mean([Y_test[i].item() >= NN_MF[i, 0] and Y_test[i].item() <= NN_MF[i, 1] for i in range(Y_test.shape[0])])
    
    PENN_II = train_test_best_model(PENN, Z_train=Z_II_train, Z_val=Z_II_val, Z_cal=Z_II_cal, Z_test=Z_II_test, Y_train=Y_train, Y_val=Y_val, Y_cal=Y_cal, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, Omega_train=Omega_train, Omega_val=Omega_val, Omega_cal=Omega_cal, Omega_test=Omega_test, lr=0.001)
    PENN_II_width[iter] = np.mean(PENN_II[:,1] - PENN_II[:,0])
    PENN_II_coverage[iter] = np.mean([Y_test[i].item() >= PENN_II[i, 0] and Y_test[i].item() <= PENN_II[i, 1] for i in range(Y_test.shape[0])])
    

    NN_II = train_test_best_model(NN, Z_train=Z_II_train, Z_val=Z_II_val, Z_cal=Z_II_cal, Z_test=Z_II_test, Y_train=Y_train, Y_val=Y_val, Y_cal=Y_cal, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, lr=0.001)
    NN_II_width[iter] = np.mean(NN_II[:,1] - NN_II[:,0])
    NN_II_coverage[iter] = np.mean([Y_test[i].item() >= NN_II[i, 0] and Y_test[i].item() <= NN_II[i, 1] for i in range(Y_test.shape[0])])


    # Write output to txt file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "output.txt")
    with open(file_path, 'w') as file:
        file.write(f"PENN_MI = c({", ".join(str(item) for item in PENN_MI_width)}) \n")
        file.write('\n')
        file.write(f"NN_MI = c({", ".join(str(item) for item in NN_MI_width)}) \n")
        file.write('\n')
        file.write(f"PENN_MF = c({", ".join(str(item) for item in PENN_MF_width)}) \n")
        file.write('\n')
        file.write(f"NN_MF = c({", ".join(str(item) for item in NN_MF_width)}) \n")
        file.write('\n')
        file.write(f"PENN_II = c({", ".join(str(item) for item in PENN_II_width)}) \n")
        file.write('\n')
        file.write(f"NN_II = c({", ".join(str(item) for item in NN_II_width)}) \n")

        file.write('\n')
        file.write('\n')
        file.write(f"PENN_MI = c({", ".join(str(item) for item in PENN_MI_coverage)}) \n")
        file.write('\n')
        file.write(f"NN_MI = c({", ".join(str(item) for item in NN_MI_coverage)}) \n")
        file.write('\n')
        file.write(f"PENN_MF = c({", ".join(str(item) for item in PENN_MF_coverage)}) \n")
        file.write('\n')
        file.write(f"NN_MF = c({", ".join(str(item) for item in NN_MF_coverage)}) \n")
        file.write('\n')
        file.write(f"PENN_II = c({", ".join(str(item) for item in PENN_II_coverage)}) \n")
        file.write('\n')
        file.write(f"NN_II = c({", ".join(str(item) for item in NN_II_coverage)}) \n")