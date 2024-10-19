import numpy as np
import torch
import pandas as pd

import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from missforest import MissForest
from sklearn.ensemble import RandomForestRegressor
import warnings

n = 2*10**4
d = 10

lr = 0.01
l1_lambda = 0
epochs = 60

def reg_func(x):
    # out = (4 * x[1] - 2)**2 + 2 * np.sin(np.pi * x[2]) * (np.sqrt(x[3]) + 1) + 6 * np.abs(x[3] - 0.5)
    out = 2 * x[1]**(1/3) + np.exp((x[2] + x[3])/2) + (4 * x[3] - 2)**2   # Bayes risk = 0.8717
    # out = 2 * x[1]**(1/3) + (4 * x[2] - 2)**2
    return out


nrep = 20
ER_PA = np.zeros(nrep)
ER_MI = np.zeros(nrep)
ER_MF = np.zeros(nrep)

for rep in range(nrep):
    X = np.random.uniform(0, 1, (n, d))
    epsilon = np.random.normal(0, 0.25, n)
    Y = np.zeros(n)
    Y_true = np.zeros(n)
    for i in range(n):
        Y_true[i] = reg_func(X[i,:])
        Y[i] = Y_true[i] + epsilon[i]

    Omega = np.random.binomial(1, 0.5, (n, d))
    # for i in range(n):
    #     if X[i,1] <= 0.5:
    #         Omega[i,1] = 1
    #     else:
    #         Omega[i,1] = 0
    sample_mean = np.sum(X*Omega, axis = 0) / np.sum(Omega, axis = 0)
    Z_ZI = X * Omega
    Z_MI = X * Omega + sample_mean * (1 - Omega)
    Z_RI = X * Omega + (1 - Omega) * np.random.uniform(0, 1, (n, d))

    Z_ZI_train = Z_ZI[0:int(n/2), :]
    Z_ZI_test = Z_ZI[int(n/2):n, :]
    Z_MI_train = Z_MI[0:int(n/2), :]
    Z_MI_test = Z_MI[int(n/2):n, :]
    Omega_train = Omega[0:int(n/2), :]
    Omega_test = Omega[int(n/2):n, :]

    Y_train = Y[0:int(n/2)]
    Y_test = Y[int(n/2):n]
    Y_true_test = Y_true[int(n/2):n]

    Z_ZI_train = torch.tensor(Z_ZI_train, dtype=torch.float32)
    Z_ZI_test = torch.tensor(Z_ZI_test, dtype=torch.float32)
    Z_MI_train = torch.tensor(Z_MI_train, dtype=torch.float32)
    Z_MI_test = torch.tensor(Z_MI_test, dtype=torch.float32)
    Omega_train = torch.tensor(Omega_train, dtype=torch.float32)
    Omega_test = torch.tensor(Omega_test, dtype=torch.float32)
    Z_Omega_train = torch.cat((Z_ZI_train, Omega_train), dim = 1)
    Z_Omega_test = torch.cat((Z_ZI_test, Omega_test), dim = 1)

    Y_train = torch.tensor(Y_train.reshape(-1, 1), dtype=torch.float32)
    Y_test = torch.tensor(Y_test.reshape(-1, 1), dtype=torch.float32)
    Y_true_test = torch.tensor(Y_true_test.reshape(-1, 1), dtype=torch.float32)


    ### pattern augmented NN
    class PANN(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch1 = nn.Sequential(
                nn.Linear(2*d, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            out = self.arch1(x)
            return out
        
    model_PA = PANN()
    PA_train_data = TensorDataset(Z_Omega_train, Y_train)
    PA_train_loader = DataLoader(dataset = PA_train_data, batch_size=20, shuffle=True)

    optimizer = optim.SGD(model_PA.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):

        for x_batch, y_batch in PA_train_loader:
            y_batch = y_batch.view(-1, 1)
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

        # if epoch % 10 == 9:
        #     print(f'Epoch {epoch}, Loss: {loss.item()}')

    ER_PA[rep] = loss_fn(model_PA(Z_Omega_test), Y_true_test) - 0.8642027



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
                nn.Linear(16, 1),
            )

        def forward(self, x):
            out = self.arch1(x)
            return out
        
    model_MI = MINN()
    train_data = TensorDataset(Z_MI_train, Y_train)
    train_loader = DataLoader(dataset = train_data, batch_size=20, shuffle=True)

    optimizer = optim.SGD(model_MI.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):

        for x_batch, y_batch in train_loader:
            y_batch = y_batch.view(-1, 1)
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

        # if epoch % 10 == 9:
        #     print(f'Epoch {epoch}, Loss: {loss.item()}')

    ER_MI[rep] = loss_fn(model_MI(Z_MI_test), Y_true_test) - 0.8642027


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
                nn.Linear(16, 1),
            )

        def forward(self, x):
            out = self.arch1(x)
            return out
    
    model_MF = MFNN()
    train_data = TensorDataset(Z_MF_train, Y_train)
    train_loader = DataLoader(dataset = train_data, batch_size=20, shuffle=True)

    optimizer = optim.SGD(model_MF.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):

        for x_batch, y_batch in train_loader:
            y_batch = y_batch.view(-1, 1)
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

        # if epoch % 10 == 9:
        #     print(f'Epoch {epoch}, Loss: {loss.item()}')

    ER_MF[rep] = loss_fn(model_MF(Z_MF_test), Y_true_test) - 0.8642027


    print(f'iteration {rep}: PA={ER_PA[rep]}, MI={ER_MI[rep]}, MF={ER_MF[rep]}')



print(f'PA={np.array2string(ER_PA, separator=', ')}')
print(f'MI={np.array2string(ER_MI, separator=', ')}')
print(f'MF={np.array2string(ER_MF, separator=', ')}')


