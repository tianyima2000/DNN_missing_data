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
import warnings

n = 10*10**4
d = 50

lr = 0.01
l1_lambda = 0
epochs = 60

def reg_func(x):
    # out = 1 / (1 + np.exp(50*(x[2]-0.7)**2 - 50*np.abs(x[5]-0.7) * np.abs(x[9]-0.7))) 
    out = 1 / (1 + np.exp(-10*x[1]**2 + 10*x[2]*x[3] + 2*x[3]))   
    return out

nrep = 20
ER_Complete = np.zeros(nrep)
ER_PA = np.zeros(nrep)
ER_MI = np.zeros(nrep)
ER_MF = np.zeros(nrep)
ER_MICE = np.zeros(nrep)

for rep in range(nrep):
    X = np.random.uniform(0, 1, (n, d))
    for i in range(n):
        X[i,1] = np.sqrt(X[i,4] * X[i,5]) + np.random.uniform(-0.1, 0.1, 1)[0]
        X[i,2] = np.sqrt(X[i,5]) + np.random.uniform(-0.1, 0.1, 1)[0]
    Y = np.zeros(n)
    probs = np.zeros(n)

    for i in range(n):
        probs[i] = reg_func(X[i,:])
        Y[i] = np.random.binomial(1, probs[i], 1)[0]

    Omega = np.random.binomial(1, 0.5, (n, d))  
    Omega[:,(3,4,5)] = np.ones((n,3))
    # for i in range(n):
    #     if probs[i] <= 0.6:
    #         Omega[i,2] = 0
    #     else:
    #         Omega[i,2] = 1
    sample_mean = np.sum(X*Omega, axis = 0) / np.sum(Omega, axis = 0)
    Z_ZI = X * Omega
    Z_MI = X * Omega + sample_mean * (1 - Omega)

    X_train = X[0:int(n/2), :]
    X_test = X[int(n/2):n, :]
    Z_ZI_train = Z_ZI[0:int(n/2), :]
    Z_ZI_test = Z_ZI[int(n/2):n, :]
    Z_MI_train = Z_MI[0:int(n/2), :]
    Z_MI_test = Z_MI[int(n/2):n, :]
    Omega_train = Omega[0:int(n/2), :]
    Omega_test = Omega[int(n/2):n, :]

    Y_train = Y[0:int(n/2)]
    Y_test = Y[int(n/2):n]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Z_ZI_train = torch.tensor(Z_ZI_train, dtype=torch.float32)
    Z_ZI_test = torch.tensor(Z_ZI_test, dtype=torch.float32)
    Z_MI_train = torch.tensor(Z_MI_train, dtype=torch.float32)
    Z_MI_test = torch.tensor(Z_MI_test, dtype=torch.float32)
    Omega_train = torch.tensor(Omega_train, dtype=torch.float32)
    Omega_test = torch.tensor(Omega_test, dtype=torch.float32)
    Z_Omega_train = torch.cat((Z_ZI_train, Omega_train), dim = 1)
    Z_Omega_test = torch.cat((Z_ZI_test, Omega_test), dim = 1)

    Y_train = torch.tensor(Y_train, dtype=torch.long)



    ### Complete data
    class CompleteNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch1 = nn.Sequential(
                nn.Linear(d, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 2),
            )

        def forward(self, x):
            out = self.arch1(x)
            return out
    
    model_Complete = CompleteNN()
    Complete_train_data = TensorDataset(X_train, Y_train)
    Complete_train_loader = DataLoader(dataset = Complete_train_data, batch_size=20, shuffle=True)

    optimizer = optim.SGD(model_Complete.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        for x_batch, y_batch in Complete_train_loader:
            optimizer.zero_grad()
            pred = model_Complete(x_batch)
            loss = loss_fn(pred, y_batch)

            # L1 penalty
            l1_penalty = 0
            for param in model_Complete.parameters():
                l1_penalty += torch.sum(torch.abs(param))
            # Add L1 penalty to the loss
            loss += l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()

    _, labels = torch.max(model_Complete(X_test), dim=1)
    pred = labels.detach().numpy()
    ER_Complete[rep] = np.mean(np.abs(pred - Y_test))



    ### pattern augmented NN
    class PANN(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch1 = nn.Sequential(
                nn.Linear(2*d, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 2),
            )

        def forward(self, x):
            out = self.arch1(x)
            return out
    
    model_PA = PANN()
    PA_train_data = TensorDataset(Z_Omega_train, Y_train)
    PA_train_loader = DataLoader(dataset = PA_train_data, batch_size=20, shuffle=True)

    optimizer = optim.SGD(model_PA.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        for x_batch, y_batch in PA_train_loader:
            optimizer.zero_grad()
            pred = model_PA(x_batch)
            loss = loss_fn(pred, y_batch)

            # L1 penalty
            l1_penalty = 0
            for param in model_PA.parameters():
                l1_penalty += torch.sum(torch.abs(param))
            # Add L1 penalty to the loss
            loss += l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()

    _, labels = torch.max(model_PA(Z_Omega_test), dim=1)
    pred = labels.detach().numpy()
    ER_PA[rep] = np.mean(np.abs(pred - Y_test))
    

    ### mean imputation
    class MINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch1 = nn.Sequential(
                nn.Linear(d, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 2),
            )

        def forward(self, x):
            out = self.arch1(x)
            return out
        
    model_MI = MINN()
    train_data = TensorDataset(Z_MI_train, Y_train)
    train_loader = DataLoader(dataset = train_data, batch_size=20, shuffle=True)

    optimizer = optim.SGD(model_MI.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model_MI(x_batch)
            loss = loss_fn(pred, y_batch)

            # L1 penalty
            l1_penalty = 0
            for param in model_MI.parameters():
                l1_penalty += torch.sum(torch.abs(param))
            # Add L1 penalty to the loss
            loss += l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()

    _, labels = torch.max(model_MI(Z_MI_test), dim=1)
    pred = labels.detach().numpy()
    ER_MI[rep] = np.mean(np.abs(pred - Y_test))
    

    ### MissForest imputation
    Z_nan = np.copy(Z_ZI)
    for i in range(n):
        for j in range(d):
            if Omega[i, j] == 0:
                Z_nan[i,j] = np.nan
    Z_nan_train = pd.DataFrame(Z_nan[0:int(n/2), :])
    Z_nan_test = pd.DataFrame(Z_nan[int(n/2):n, :])
    rgr = RandomForestRegressor(n_jobs=-1)
    warnings.filterwarnings('ignore')
    mf = MissForest(rgr)
    mf.fit(x=Z_nan_train)
    Z_MF_train = mf.transform(Z_nan_train)
    Z_MF_test = mf.transform(Z_nan_test)
    Z_MF_train = Z_MF_train.to_numpy()
    Z_MF_test = Z_MF_test.to_numpy()

    Z_MF_train = torch.tensor(Z_MF_train, dtype=torch.float32)
    Z_MF_test = torch.tensor(Z_MF_test, dtype=torch.float32)

    ### MissForest imputation
    class MFNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch1 = nn.Sequential(
                nn.Linear(d, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 2),
            )

        def forward(self, x):
            out = self.arch1(x)
            return out
        
    model_MF = MFNN()
    train_data = TensorDataset(Z_MF_train, Y_train)
    train_loader = DataLoader(dataset = train_data, batch_size=20, shuffle=True)

    optimizer = optim.SGD(model_MF.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model_MF(x_batch)
            loss = loss_fn(pred, y_batch)

            # L1 penalty
            l1_penalty = 0
            for param in model_MF.parameters():
                l1_penalty += torch.sum(torch.abs(param))
            # Add L1 penalty to the loss
            loss += l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()

    _, labels = torch.max(model_MF(Z_MF_test), dim=1)
    pred = labels.detach().numpy()
    ER_MF[rep] = np.mean(np.abs(pred - Y_test))


    ### MICE imputation
    Z_nan = np.copy(Z_ZI)
    for i in range(n):
        for j in range(d):
            if Omega[i, j] == 0:
                Z_nan[i,j] = np.nan
    Z_nan_train = pd.DataFrame(Z_nan[0:int(n/2), :])
    Z_nan_test = pd.DataFrame(Z_nan[int(n/2):n, :])

    warnings.filterwarnings('ignore')
    imputer = IterativeImputer(max_iter=10, random_state=0)
    fitted_imputer = imputer.fit(Z_nan_train)
    Z_MICE_train = fitted_imputer.transform(Z_nan_train)
    Z_MICE_test = fitted_imputer.transform(Z_nan_test)
    Z_MICE_train = Z_MICE_train
    Z_MICE_test = Z_MICE_test

    Z_MICE_train = torch.tensor(Z_MICE_train, dtype=torch.float32)
    Z_MICE_test = torch.tensor(Z_MICE_test, dtype=torch.float32)

    class MICENN(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch1 = nn.Sequential(
                nn.Linear(d, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 2),
            )

        def forward(self, x):
            out = self.arch1(x)
            return out
        
    model_MICE = MICENN()
    train_data = TensorDataset(Z_MICE_train, Y_train)
    train_loader = DataLoader(dataset = train_data, batch_size=20, shuffle=True)

    optimizer = optim.SGD(model_MICE.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model_MICE(x_batch)
            loss = loss_fn(pred, y_batch)

            # L1 penalty
            l1_penalty = 0
            for param in model_MICE.parameters():
                l1_penalty += torch.sum(torch.abs(param))
            # Add L1 penalty to the loss
            loss += l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()

    _, labels = torch.max(model_MICE(Z_MICE_test), dim=1)
    pred = labels.detach().numpy()
    ER_MICE[rep] = np.mean(np.abs(pred - Y_test))
    
    print(f'iteration {rep}: Complete={ER_Complete[rep]} PA={ER_PA[rep]}, MI={ER_MI[rep]}, MF={ER_MF[rep]}, MICE={ER_MICE[rep]}')


ER_Complete_txt = np.array2string(ER_Complete, separator=', ')
ER_PA_txt = np.array2string(ER_PA, separator=', ')
ER_MI_txt = np.array2string(ER_MI, separator=', ')
ER_MF_txt = np.array2string(ER_MF, separator=', ')
ER_MICE_txt = np.array2string(ER_MICE, separator=', ')


script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = f"classification3-d{d}-output.txt"
file_path = os.path.join(script_dir, file_name)

with open(file_path, 'w') as file:
    # Write text to the file
    file.write(f'ER_Complete={ER_Complete_txt}\n')
    file.write(f'ER_PA={ER_PA_txt}\n')
    file.write(f'ER_MI={ER_MI_txt}\n')
    file.write(f'ER_MF={ER_MF_txt}\n')
    file.write(f'ER_MICE={ER_MICE_txt}\n')