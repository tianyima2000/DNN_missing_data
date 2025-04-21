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
from torchvision import datasets

from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        train_loader = DataLoader(dataset = train_data, batch_size=64, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()
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
        train_loader = DataLoader(dataset = train_data, batch_size=64, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()
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
        val_loss[i] = output["val_loss"]
        
    min_index = np.argmin(val_loss)
    return test_loss[min_index]



class PENN(nn.Module):
    def __init__(self):
        super().__init__()
        embedding_dim = 3
        
        # Construct the neural network f1
        self.f1 = nn.Sequential(
            nn.Linear(28*28, 100),  
            nn.ReLU(),
            nn.Linear(100, 100),  
            nn.ReLU(),
            nn.Linear(100, 100),  
            nn.ReLU(),
        )

        # Construct the neural network f2, i.e. the embedding function
        self.f2 = nn.Sequential(
            nn.Linear(7*7, 30),  
            nn.ReLU(),
            nn.Linear(30, 30),  
            nn.ReLU(),
            nn.Linear(30, embedding_dim)
        )

        
        # Construct the neural network f3
        self.f3 = nn.Sequential(
            nn.Linear(100 + embedding_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),  
            nn.ReLU(),
            nn.Linear(100, 10)  
        )
    
    # Combine f1, f2 and f3 to construct the Pattern Embedding Neural Network (PENN)
    def forward(self, z, omega):
        z = z.view(-1, 28*28)
        omega = omega.view(-1, 7*7)

        # compute the output of f1 and f2
        f1_output = self.f1(z)
        f2_output = self.f2(omega)
        
        # Concatenate the output of f1 and f2
        combined_features = torch.cat((f1_output, f2_output), dim=1)
        
        # Apply the combined network
        final_output = self.f3(combined_features)
        
        return final_output
    


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100,10)
        )
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        final_output = self.nn(x)
        return final_output
    


train_data = datasets.MNIST('./data', train=True, download=True)
X_train = train_data.data.float() / 255
Y_train = train_data.targets.to(torch.long)

test_data = datasets.MNIST('./data', train=False, download=True)
X_test = test_data.data.float() / 255
Y_test = test_data.targets.to(torch.long)

X = torch.cat((X_train, X_test), dim=0).to(device)
Y = torch.cat((Y_train, Y_test), dim=0).to(device)


total_iterations = 50
PENN_loss = np.zeros(total_iterations)
NN_loss = np.zeros(total_iterations)

for iter in tqdm(range(total_iterations), bar_format='[{elapsed}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}'):
    Omega = torch.tensor(np.random.binomial(1, 0.7, (70000, 7, 7)), dtype=torch.float32).to(device)
    Omega_mask = torch.tensor(np.zeros((70000, 28, 28)), dtype=torch.float32).to(device)

    for j in range(7):
        for k in range(7):
            Omega_mask[:, j*4:(j+1)*4, k*4:(k+1)*4] = Omega[:, j, k].unsqueeze(-1).unsqueeze(-1)

    Z = X*Omega_mask

    Z_train, Z_test, Omega_train, Omega_test, Y_train, Y_test = train_test_split(Z, Omega, Y, test_size=0.1)
    Z_train, Z_val, Omega_train, Omega_val, Y_train, Y_val = train_test_split(Z_train, Omega_train, Y_train, test_size=0.11111)

    prune_amount_vec = [0.9, 0.8, 0.6, 0.2]

    PENN_loss[iter] = train_test_best_model(PENN, Z_train=Z_train, Z_val=Z_val, Z_test=Z_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, Omega_train=Omega_train, Omega_val=Omega_val, Omega_test=Omega_test, lr=0.001, weight_decay=0)
    NN_loss[iter] = train_test_best_model(NN, Z_train=Z_train, Z_val=Z_val, Z_test=Z_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test, 
                                               prune_amount_vec=prune_amount_vec, lr=0.001, weight_decay=0)
    
    # Write output to txt file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "output_MCAR.txt")
    with open(file_path, 'w') as file:
        file.write(f"PENN = c({", ".join(str(item) for item in PENN_loss)}) \n")
        file.write('\n')
        file.write(f"NN = c({", ".join(str(item) for item in NN_loss)}) \n")
        file.write('\n')