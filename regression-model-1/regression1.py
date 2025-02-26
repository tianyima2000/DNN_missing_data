import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import pandas as pd

from IPython.display import clear_output

import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.utils.prune as prune

from missforest.missforest import MissForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer



"""
The following function globally prunes all weights (not including the bias vectors) 
in nn.Linear layers in the model using L1 unstructured pruning,
reinitializes the surviving weights using Kaiming uniform initialization,
and installs a forward pre-hook to enforce that the pruned weights remain zero.

Inputs:
    model (nn.Module): The model to prune.
    amount (float): Fraction of weights to prune globally (e.g., 0.8 for 80%).

Output: The pruned model.
"""
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
    Z: covariate matrix (n times d), including training, validation and testing data
    Y: response vector (n-dimensional), including training, validation and testing data
    lr: learning rate for Adam optimiser
    prune_amount: the proportion of weights to be set to zero
    Omega: If Omega=None, then a standard neural network is used. 
        If Omega is the observation pattern matrix (n times d), then PENN is used.
    epochs: maximum number of epochs to train
    weight_decay: weight decay for Adam optimiser
    prune_start: the epoch that pruning starts
    patience: patience for early stopping
    live_plot: if True, then the training loss and validation losses will be ploted live

Output: testing loss and validation loss
"""
def train_test_model(model, Z, Y, lr, prune_amount, Omega=None, epochs=200, weight_decay=0.001, prune_start=10, patience=10, live_plot=False):

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
    if Omega is not None:
        ### split data into 50% training data, 25% validation data, 25% testing data
        Z_train = Z[0:int(n/2), :]
        Z_val = Z[int(n/2):int(3*n/4), :]
        Z_test = Z[int(3*n/4):n, :]
        Omega_train = Omega[0:int(n/2), :]
        Omega_val = Omega[int(n/2):int(3*n/4), :]
        Omega_test = Omega[int(3*n/4):n, :]

        Y_train = Y[0:int(n/2)]
        Y_val = Y[int(n/2):int(3*n/4)]
        Y_test = Y[int(3*n/4):n]

        Z_train = torch.tensor(Z_train, dtype=torch.float32)
        Z_val = torch.tensor(Z_val, dtype=torch.float32)
        Z_test = torch.tensor(Z_test, dtype=torch.float32)
        Omega_train = torch.tensor(Omega_train, dtype=torch.float32)
        Omega_val = torch.tensor(Omega_val, dtype=torch.float32)
        Omega_test = torch.tensor(Omega_test, dtype=torch.float32)

        Y_train = torch.tensor(Y_train.reshape(-1, 1), dtype=torch.float32)
        Y_val = torch.tensor(Y_val.reshape(-1, 1), dtype=torch.float32)
        Y_test = torch.tensor(Y_test.reshape(-1, 1), dtype=torch.float32)

        ### set up for training
        train_data = TensorDataset(Z_train, Omega_train, Y_train)
        train_loader = DataLoader(dataset = train_data, batch_size=200, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

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

            # Early stopping
            if epoch >= prune_start:
                min_index = val_losses.index(min(val_losses[prune_start:]))
                if epoch - min_index >= patience:
                    break
             

        # close the plot
        if live_plot:
            plt.ioff()
            plt.close(fig)   
        
        ### return testing loss and validation loss as a dictionary
        model.eval()
        return {"test_loss": loss_fn(model(Z_test, Omega_test), Y_test), "val_loss": loss_fn(model(Z_val, Omega_val), Y_val)}


    ##### if Omega is None
    if Omega is None:
        ### split data into 50% training data, 25% validation data, 25% testing data
        Z_train = Z[0:int(n/2), :]
        Z_val = Z[int(n/2):int(3*n/4), :]
        Z_test = Z[int(3*n/4):n, :]

        Y_train = Y[0:int(n/2)]
        Y_val = Y[int(n/2):int(3*n/4)]
        Y_test = Y[int(3*n/4):n]

        Z_train = torch.tensor(Z_train, dtype=torch.float32)
        Z_val = torch.tensor(Z_val, dtype=torch.float32)
        Z_test = torch.tensor(Z_test, dtype=torch.float32)

        Y_train = torch.tensor(Y_train.reshape(-1, 1), dtype=torch.float32)
        Y_val = torch.tensor(Y_val.reshape(-1, 1), dtype=torch.float32)
        Y_test = torch.tensor(Y_test.reshape(-1, 1), dtype=torch.float32)

        ### set up for training
        train_data = TensorDataset(Z_train, Y_train)
        train_loader = DataLoader(dataset = train_data, batch_size=200, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

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

            # Early stopping
            if epoch >= prune_start:
                min_index = val_losses.index(min(val_losses[prune_start:]))
                if epoch - min_index >= patience:
                    break
    

        # close the plot
        if live_plot:
            plt.ioff()
            plt.close(fig)   
        
        ### return testing loss and validation loss as a dictionary
        model.eval()
        return {"test_loss": loss_fn(model(Z_test), Y_test), "val_loss": loss_fn(model(Z_val), Y_val)}
    

def train_test_best_model(model_class, Z, Y, lr, prune_amount_vec, Omega=None, epochs=200, weight_decay=0.001, prune_start=10, patience=10, live_plot=False):
    N = len(prune_amount_vec)
    test_loss = np.zeros(N)
    val_loss = np.zeros(N)
    for i in range(N):
        model = model_class()
        output = train_test_model(model=model, Z=Z, Y=Y, lr=lr, prune_amount=prune_amount_vec[i], Omega=Omega, 
                                  epochs=epochs, weight_decay=weight_decay, prune_start=prune_start, patience=patience, live_plot=live_plot)
        test_loss[i] = output["test_loss"]
        val_loss[i] = output["val_loss"]
        
    min_index = np.argmin(val_loss)
    return test_loss[min_index]



### Pattern Embedding Neural Network (PENN)
class PENN(nn.Module):
    def __init__(self):
        super().__init__()
        embedding_dim = 5
        
        # Construct the neural network f1
        self.f1 = nn.Sequential(
            nn.Linear(d, 70),  
            nn.ReLU(),
            nn.Linear(70, 70),  
            nn.ReLU(),
            nn.Linear(70, 70),  
            nn.ReLU()
        )

        # Construct the neural network f2, i.e. the embedding function
        self.f2 = nn.Sequential(
            nn.Linear(d, 30),  
            nn.ReLU(),
            nn.Linear(30, 30),  
            nn.ReLU(),
            nn.Linear(30, 30),  
            nn.ReLU(),
            nn.Linear(30, embedding_dim)
        )

        
        # Construct the neural network f3
        self.f3 = nn.Sequential(
            nn.Linear(70 + embedding_dim, 70),
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 1)  
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
            nn.Linear(d, 70),
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70, 70),
            nn.ReLU(),
            nn.Linear(70,1)
        )
    
    def forward(self, x):
        final_output = self.nn(x)
        return final_output
    


total_iterations = 1
PENN_ZI_loss = np.zeros(total_iterations)
NN_ZI_loss = np.zeros(total_iterations)
PENN_MF_loss = np.zeros(total_iterations)
NN_MF_loss = np.zeros(total_iterations)
PENN_MICE_loss = np.zeros(total_iterations)
NN_MICE_loss = np.zeros(total_iterations)

for iter in tqdm(range(total_iterations), bar_format='[{elapsed}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}'):
    n = 2*10**4    # This includes 50% training data, 25% validationm data and 25% testing data. 
    d = 20
    sigma = 0.5

    # Define regression function
    def reg_func(x):
        y = np.exp(x[1] + x[2]) + 4 * x[3]**2    # Bayes risk approx 1.448 + sigma^2
        return y

    # Generate X and Y
    X = np.random.uniform(-1 , 1, (n, d))
    epsilon = np.random.normal(0, sigma, n)
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = reg_func(X[i,:]) + epsilon[i]

    # Generate Omega, which has iid Ber(0.5) coordinates, independent of X
    Omega = np.random.binomial(1, 0.5, (n, d))

    # Z_nan is the incomplete dataset with missing entries given by np.nan
    Z_nan = np.copy(X)
    for i in range(n):
        for j in range(d):
            if Omega[i, j] == 0:
                Z_nan[i,j] = np.nan

    # Z_ZI is the zero imputed data set
    Z_ZI = X * Omega

    # Missforest imputation
    warnings.filterwarnings("ignore")
    Z_nan_df = pd.DataFrame(Z_nan)
    rgr = RandomForestRegressor(n_jobs=-1)
    mf_imputer = MissForest(rgr, verbose=False)
    Z_MF = mf_imputer.fit_transform(Z_nan_df)
    Z_MF = Z_MF.to_numpy()

    # Mice imputation
    mice_imputer = IterativeImputer(max_iter=10)
    Z_MICE = mice_imputer.fit_transform(Z_nan)

    prune_amount_vec = [0.9, 0.8, 0.7, 0.5, 0.2]

    PENN_ZI_loss[iter] = train_test_best_model(model_class=PENN, Z=Z_ZI, Omega=Omega, Y=Y, lr=0.001, prune_amount_vec=prune_amount_vec, prune_start=10, patience = 10) - 1.448 - sigma**2
    NN_ZI_loss[iter] = train_test_best_model(model_class=NN, Z=Z_ZI, Y=Y, lr=0.001, prune_amount_vec=prune_amount_vec, prune_start=10, patience = 10) - 1.448 - sigma**2

    PENN_MF_loss[iter] = train_test_best_model(model_class=PENN, Z=Z_MF, Omega=Omega, Y=Y, lr=0.001, prune_amount_vec=prune_amount_vec, prune_start=10, patience = 10) - 1.448 - sigma**2
    NN_MF_loss[iter] = train_test_best_model(model_class=NN, Z=Z_MF, Y=Y, lr=0.001, prune_amount_vec=prune_amount_vec, prune_start=10, patience = 10) - 1.448 - sigma**2

    PENN_MICE_loss[iter] = train_test_best_model(model_class=PENN, Z=Z_MICE, Omega=Omega, Y=Y, lr=0.001, prune_amount_vec=prune_amount_vec, prune_start=10, patience = 10) - 1.448 - sigma**2
    NN_MICE_loss[iter] = train_test_best_model(model_class=NN, Z=Z_MICE, Y=Y, lr=0.001, prune_amount_vec=prune_amount_vec, prune_start=10, patience = 10) - 1.448 - sigma**2



print(f"PENN_ZI_loss: {", ".join(str(item) for item in PENN_ZI_loss)}")
print(f"NN_ZI_loss: {", ".join(str(item) for item in NN_ZI_loss)}")

print(f"PENN_MF_loss: {", ".join(str(item) for item in PENN_MF_loss)}")
print(f"NN_MF_loss: {", ".join(str(item) for item in NN_MF_loss)}")

print(f"PENN_MICE_loss: {", ".join(str(item) for item in PENN_MICE_loss)}")
print(f"NN_MICE_loss: {", ".join(str(item) for item in NN_MICE_loss)}")