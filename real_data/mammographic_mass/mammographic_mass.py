import numpy as np
import torch
import pandas as pd
import os

import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from missforest import MissForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
mammographic_mass = fetch_ucirepo(id=161) 
  
# data (as pandas dataframes) 
X = mammographic_mass.data.features 
Y = mammographic_mass.data.targets 


Omega = X.isna().to_numpy()
Omega = Omega.astype(int)

X_ZI = X.fillna(0)
X_ZI = X_ZI.to_numpy()
scaler = StandardScaler()
scaler.fit(X_ZI)
X_ZI = scaler.transform(X_ZI)

sample_mean = X.mean()
X_MI = X.fillna(sample_mean)
scaler = StandardScaler()
scaler.fit(X_MI)
X_MI = scaler.transform(X_MI)

Y = Y.to_numpy().reshape(-1)


nrep = 20
ER_PE = np.zeros(nrep)
ER_MI = np.zeros(nrep)
for rep in range(nrep):
    X_ZI_train, X_ZI_test, X_MI_train, X_MI_test, Y_train, Y_test, Omega_train, Omega_test = train_test_split(X_ZI, X_MI, Y, Omega, test_size=161, random_state=42)
    X_ZI_train = torch.tensor(X_ZI_train, dtype=torch.float32)
    X_ZI_test = torch.tensor(X_ZI_test, dtype=torch.float32)

    X_MI_train = torch.tensor(X_MI_train, dtype=torch.float32)
    X_MI_test = torch.tensor(X_MI_test, dtype=torch.float32)

    Omega_train = torch.tensor(Omega_train, dtype=torch.float32)
    Omega_test = torch.tensor(Omega_test, dtype=torch.float32)

    Y_train = torch.tensor(Y_train, dtype=torch.long)

    ### PE
    lr = 0.001
    epochs = 300

    class TwoStreamModel(nn.Module):
        def __init__(self):
            super().__init__()
            embedding_dim = 2
            
            # The embedding layer
            self.embedding = nn.Sequential(
                nn.Linear(5, 2),  
                nn.ReLU(),
                # nn.Linear(5, embedding_dim),  
                # nn.ReLU()
            )
            
            # Combined network for the layers after embedding
            self.combined = nn.Sequential(
                nn.Linear(5 + embedding_dim, 5),
                nn.ReLU(),
                nn.Linear(5, 5),
                nn.ReLU(),
                nn.Linear(5, 5),
                nn.ReLU(),
                nn.Linear(5, 2)  # Output layer
            )
        
        def forward(self, x, omega):
            # Apply the embedding layer to omega
            embedded = self.embedding(omega)
            
            # Concatenate the embedding with x
            combined_features = torch.cat((x, embedded), dim=1)
            
            # Apply the combined network
            final_out = self.combined(combined_features)
            
            return final_out
        

    model_PE = TwoStreamModel()
    PE_train_data = TensorDataset(X_MI_train, Omega_train, Y_train)
    PE_train_loader = DataLoader(dataset = PE_train_data, batch_size=32, shuffle=True)

    optimizer = optim.Adam(model_PE.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()   

    for epoch in range(epochs):

        for x_batch, omega_batch, y_batch in PE_train_loader:
            optimizer.zero_grad()
            pred = model_PE(x_batch, omega_batch)
            loss = loss_fn(pred, y_batch)

            loss.backward()
            optimizer.step()

        scheduler.step()

    _, labels = torch.max(model_PE(X_MI_test, Omega_test), dim=1)
    pred = labels.detach().numpy()
    ER_PE[rep] = np.mean(pred != Y_test)


    ### MI
    lr = 0.001
    epochs = 300

    class FCModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(5, 5),
                nn.ReLU(),
                nn.Linear(5, 5),
                nn.ReLU(),
                nn.Linear(5, 5),
                nn.ReLU(),
                nn.Linear(5, 2)  # Output layer
            )
        
        def forward(self, x):   
            final_out = self.fc(x)
            return final_out
        

    model_MI = FCModel()
    MI_train_data = TensorDataset(X_MI_train, Y_train)
    MI_train_loader = DataLoader(dataset = MI_train_data, batch_size=32, shuffle=True)

    optimizer = optim.Adam(model_MI.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()   

    for epoch in range(epochs):

        for x_batch, y_batch in MI_train_loader:
            optimizer.zero_grad()
            pred = model_MI(x_batch)
            loss = loss_fn(pred, y_batch)

            loss.backward()
            optimizer.step()

        scheduler.step()

    _, labels = torch.max(model_MI(X_MI_test), dim=1)
    pred = labels.detach().numpy()
    ER_MI[rep] = np.mean(pred != Y_test)

    print(f'iteration {rep}: PE={ER_PE[rep]}, MI={ER_MI[rep]}')



ER_PE_txt = np.array2string(ER_PE, separator=', ')
ER_MI_txt = np.array2string(ER_MI, separator=', ')

script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = "output"
file_path = os.path.join(script_dir, file_name)

with open(file_path, 'w') as file:
    # Write text to the file
    file.write(f'ER_PE={ER_PE_txt}\n')
    file.write(f'ER_MI={ER_MI_txt}\n')