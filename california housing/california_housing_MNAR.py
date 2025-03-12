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
        loss_fn = nn.MSELoss()
        early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

        ### start training
        for epoch in range(epochs):

            model.train()
            if epoch == prune_start:
                model = global_prune(model, amount = prune_amount)
                optimizer = optim.Adam(model.parameters(), lr=lr)
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
        return {"test_loss": loss_fn(model(Z_test, Omega_test), Y_test), "val_loss": loss_fn(model(Z_val, Omega_val), Y_val)}


    ##### if Omega is None
    if Omega_train is None:

        ### set up for training
        train_data = TensorDataset(Z_train, Y_train)
        train_loader = DataLoader(dataset = train_data, batch_size=200, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

        ### start training
        for epoch in range(epochs):

            model.train()
            if epoch == prune_start:
                model = global_prune(model, amount = prune_amount)
                optimizer = optim.Adam(model.parameters(), lr=lr)
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
        return {"test_loss": loss_fn(model(Z_test), Y_test), "val_loss": loss_fn(model(Z_val), Y_val)}
    

def train_test_best_model(model_class, Z_train, Z_val, Z_test, Y_train, Y_val, Y_test, lr, prune_amount_vec, Omega_train=None, Omega_val=None, Omega_test=None, epochs=200, weight_decay=0.001, prune_start=10, patience=10, live_plot=False):
    N = len(prune_amount_vec)
    test_loss = np.zeros(N)
    val_loss = np.zeros(N)
    for i in range(N):
        model = model_class()
        output = train_test_model(model=model, Z_train=Z_train, Z_val=Z_val, Z_test=Z_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, lr=lr, 
                                  prune_amount=prune_amount_vec[i], Omega_train=Omega_train, Omega_val=Omega_val, Omega_test=Omega_test, 
                                  epochs=epochs, weight_decay=weight_decay, prune_start=prune_start, patience=patience, live_plot=live_plot)
        test_loss[i] = output["test_loss"]
        val_loss[i] = output["val_loss"]
        
    min_index = np.argmin(val_loss)
    return test_loss[min_index]



# Pattern Embedding Neural Networks (PENN)
class PENN(nn.Module):
    def __init__(self):
        super().__init__()
        inputdim = 8
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
            nn.Linear(width3, 1)  
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
        inputdim = 8
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
            nn.Linear(width,1)
        )
    
    def forward(self, x):
        final_output = self.nn(x)
        return final_output



total_iterations = 20

from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing(as_frame=True)
data = california_housing.frame
X = data.drop('MedHouseVal', axis=1)
Y = data['MedHouseVal']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

PENN_ZI_loss = np.zeros(total_iterations)
NN_ZI_loss = np.zeros(total_iterations)
PENN_MF_loss = np.zeros(total_iterations)
NN_MF_loss = np.zeros(total_iterations)
PENN_MICE_loss = np.zeros(total_iterations)
NN_MICE_loss = np.zeros(total_iterations)

for iter in tqdm(range(total_iterations), bar_format='[{elapsed}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}'):
    Omega = np.random.binomial(1, 0.7, (20640, 8))
    for i in range(20640):
        income = income = X_scaled['MedInc'][i]
        prob = 1 / (0.4*np.exp(income) + 1)
        Omega[i, 0] = np.random.binomial(1, prob, 1)[0]
    Z = X.copy(deep=True)
    Z = Z.mask(Omega==0)
    scaler = StandardScaler()
    Z = pd.DataFrame(scaler.fit_transform(Z), columns=Z.columns)
    Z_ZI = Z.fillna(0).to_numpy()

    # MissForest imputation
    warnings.filterwarnings("ignore")
    rgr = RandomForestRegressor(n_jobs=-1)
    mf_imputer = MissForest(rgr, verbose=False)
    Z_MF = mf_imputer.fit_transform(Z).to_numpy()

    # Mice imputation
    mice_imputer = IterativeImputer(max_iter=10)
    Z_MICE = mice_imputer.fit_transform(Z)

    Z_ZI_train, Z_ZI_test, Z_MF_train, Z_MF_test, Z_MICE_train, Z_MICE_test, Omega_train, Omega_test, Y_train, Y_test = train_test_split(Z_ZI, Z_MF, Z_MICE, Omega, Y, test_size=0.1)
    Z_ZI_train, Z_ZI_val, Z_MF_train, Z_MF_val, Z_MICE_train, Z_MICE_val, Omega_train, Omega_val, Y_train, Y_val = train_test_split(Z_ZI_train, Z_MF_train, Z_MICE_train, Omega_train, Y_train, test_size=0.11111)

    Z_ZI_train = torch.tensor(Z_ZI_train, dtype=torch.float32)
    Z_ZI_val = torch.tensor(Z_ZI_val, dtype=torch.float32)
    Z_ZI_test = torch.tensor(Z_ZI_test, dtype=torch.float32)

    Z_MF_train = torch.tensor(Z_MF_train, dtype=torch.float32)
    Z_MF_val = torch.tensor(Z_MF_val, dtype=torch.float32)
    Z_MF_test = torch.tensor(Z_MF_test, dtype=torch.float32)

    Z_MICE_train = torch.tensor(Z_MICE_train, dtype=torch.float32)
    Z_MICE_val = torch.tensor(Z_MICE_val, dtype=torch.float32)
    Z_MICE_test = torch.tensor(Z_MICE_test, dtype=torch.float32)

    Omega_train = torch.tensor(Omega_train, dtype=torch.float32)
    Omega_val = torch.tensor(Omega_val, dtype=torch.float32)
    Omega_test = torch.tensor(Omega_test, dtype=torch.float32)

    Y_train = torch.tensor(Y_train.to_numpy().reshape(-1,1), dtype=torch.float32)
    Y_val = torch.tensor(Y_val.to_numpy().reshape(-1,1), dtype=torch.float32)
    Y_test = torch.tensor(Y_test.to_numpy().reshape(-1,1), dtype=torch.float32)

    prune_amount_vec = [0.9, 0.8, 0.6, 0.2]

    PENN_ZI_loss[iter] = train_test_best_model(PENN, Z_train=Z_ZI_train, Z_val=Z_ZI_val, Z_test=Z_ZI_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, Omega_train=Omega_train, Omega_val=Omega_val, Omega_test=Omega_test, lr=0.001) / Y_test.var(unbiased=False)
    NN_ZI_loss[iter] = train_test_best_model(NN, Z_train=Z_ZI_train, Z_val=Z_ZI_val, Z_test=Z_ZI_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, lr=0.001) / Y_test.var(unbiased=False)
    
    PENN_MF_loss[iter] = train_test_best_model(PENN, Z_train=Z_MF_train, Z_val=Z_MF_val, Z_test=Z_MF_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, Omega_train=Omega_train, Omega_val=Omega_val, Omega_test=Omega_test, lr=0.001) / Y_test.var(unbiased=False)
    NN_MF_loss[iter] = train_test_best_model(NN, Z_train=Z_MF_train, Z_val=Z_MF_val, Z_test=Z_MF_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, lr=0.001) / Y_test.var(unbiased=False)
    
    PENN_MICE_loss[iter] = train_test_best_model(PENN, Z_train=Z_MICE_train, Z_val=Z_MICE_val, Z_test=Z_MICE_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, Omega_train=Omega_train, Omega_val=Omega_val, Omega_test=Omega_test, lr=0.001) / Y_test.var(unbiased=False)
    NN_MICE_loss[iter] = train_test_best_model(NN, Z_train=Z_MICE_train, Z_val=Z_MICE_val, Z_test=Z_MICE_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, lr=0.001) / Y_test.var(unbiased=False)
    
    # Write output to txt file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "output.txt")
    with open(file_path, 'w') as file:
        file.write(f"PENN_ZI = c({", ".join(str(item) for item in PENN_ZI_loss)}) \n")
        file.write('\n')
        file.write(f"NN_ZI = c({", ".join(str(item) for item in NN_ZI_loss)}) \n")
        file.write('\n')
        file.write(f"PENN_MF = c({", ".join(str(item) for item in PENN_MF_loss)}) \n")
        file.write('\n')
        file.write(f"NN_MF = c({", ".join(str(item) for item in NN_MF_loss)}) \n")
        file.write('\n')
        file.write(f"PENN_MICE = c({", ".join(str(item) for item in PENN_MICE_loss)}) \n")
        file.write('\n')
        file.write(f"NN_MICE = c({", ".join(str(item) for item in NN_MICE_loss)}) \n")
