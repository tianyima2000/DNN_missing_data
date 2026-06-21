import os
import random
import pickle
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.utils.prune as prune

from missforest.missforest import MissForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler


# ============================================================
# Basic setup
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ceil(x):
    return int(np.ceil(x))


# ============================================================
# Global simulation parameters
# ============================================================

n = 2 * 10**4
d = 20
sigma = 0.5

width_1 = ceil(np.sqrt(0.5 * n / 2))   # training sample size is 0.5n
width_2 = ceil(width_1 / 5)
embedding_dim = min(ceil(np.sqrt(d)), ceil(width_1 / 20))

print("Device:", device)
print("n =", n)
print("d =", d)
print("width_1 =", width_1)
print("width_2 =", width_2)
print("embedding_dim =", embedding_dim)


# ============================================================
# Early stopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss

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
        return {
            name: param.clone().detach()
            for name, param in model.state_dict().items()
        }

    def restore_weights(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# ============================================================
# Pruning
# ============================================================

def prune_and_reinit(model, amount, init_fn=nn.init.kaiming_uniform_):
    """
    Globally prune the model except for layers under model.f2.

    Therefore, the pattern embedding function f2 is not pruned.
    """

    parameters_to_prune = []

    for name, module in model.named_modules():
        if name == "f2" or name.startswith("f2."):
            continue

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    with torch.no_grad():
        for name, module in model.named_modules():
            if name == "f2" or name.startswith("f2."):
                continue

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, "weight_orig") and hasattr(module, "weight_mask"):
                    init_fn(module.weight_orig)
                    module.weight_orig.mul_(module.weight_mask)

    return model


# ============================================================
# PENN
# ============================================================

class PENN(nn.Module):
    def __init__(self):
        super().__init__()

        self.f1 = nn.Sequential(
            nn.Linear(d, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),
            nn.ReLU(),
        )

        # Pattern embedding function
        self.f2 = nn.Sequential(
            nn.Linear(d, width_2),
            nn.ReLU(),
            nn.Linear(width_2, width_2),
            nn.ReLU(),
            nn.Linear(width_2, embedding_dim),
        )

        self.f3 = nn.Sequential(
            nn.Linear(width_1 + embedding_dim, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),
            nn.ReLU(),
            nn.Linear(width_1, width_1),
            nn.ReLU(),
            nn.Linear(width_1, 1),
        )

    def forward(self, z, omega):
        f1_output = self.f1(z)
        f2_output = self.f2(omega)

        combined_features = torch.cat((f1_output, f2_output), dim=1)
        final_output = self.f3(combined_features)

        return final_output


# ============================================================
# Helper functions
# ============================================================

def reg_func(x):
    """
    Regression function from model_3.
    """
    y = np.exp(x[1] + x[2]) + 4 * x[3] ** 2
    return y


def to_float_tensor(x, response=False):
    arr = np.asarray(x, dtype=np.float32)

    if response:
        arr = arr.reshape(-1, 1)

    return torch.tensor(arr, dtype=torch.float32)


def maybe_to_numpy(x):
    """
    MissForest may return either a pandas object or a numpy array,
    depending on package version.
    """
    if hasattr(x, "to_numpy"):
        return x.to_numpy()

    return np.asarray(x)


def module_state_dict_cpu(module):
    return {
        k: v.detach().cpu().clone()
        for k, v in module.state_dict().items()
    }


# ============================================================
# Generate and impute data
# ============================================================

def generate_and_impute_data(random_state=1):
    """
    Generate the synthetic model_3 data and construct MI, MF, II datasets.
    """

    set_seed(random_state)

    # Generate X
    X = np.random.uniform(-1, 1, size=(n, d))

    # Model_3 dependence structure
    X[:, 1] = (
        np.sqrt(X[:, 4] + 1)
        - 0.7
        + np.random.uniform(-0.3, 0.3, size=n)
    )

    X[:, 3] = (
        0.7 * X[:, 5]
        + np.random.uniform(-0.3, 0.3, size=n)
    )

    # Generate Y
    epsilon = np.random.normal(0, sigma, size=n)
    Y = np.zeros(n)

    for i in range(n):
        Y[i] = reg_func(X[i, :]) + epsilon[i]

    # Generate observation pattern
    Omega = np.random.binomial(1, 0.7, size=(n, d))

    # Incomplete data
    Z_nan = np.copy(X)
    Z_nan[Omega == 0] = np.nan
    Z_nan = pd.DataFrame(Z_nan)

    # Mean imputation
    Z_MI = Z_nan.fillna(Z_nan.mean())
    scaler = StandardScaler()
    Z_MI = scaler.fit_transform(Z_MI)

    # MissForest imputation
    warnings.filterwarnings("ignore")

    rgr = RandomForestRegressor(n_jobs=1)
    mf_imputer = MissForest(rgr, verbose=False)

    Z_MF = mf_imputer.fit_transform(Z_nan)
    Z_MF = maybe_to_numpy(Z_MF)

    scaler = StandardScaler()
    Z_MF = scaler.fit_transform(Z_MF)

    # Iterative imputation
    II_imputer = IterativeImputer(max_iter=10)
    Z_II = II_imputer.fit_transform(Z_nan.to_numpy())

    scaler = StandardScaler()
    Z_II = scaler.fit_transform(Z_II)

    return {
        "X": X,
        "Y": Y,
        "Omega": Omega,
        "Z_nan": Z_nan.to_numpy(),
        "Z_MI": Z_MI,
        "Z_MF": Z_MF,
        "Z_II": Z_II,
    }


# ============================================================
# Fixed split
# ============================================================

def fixed_split(data):
    """
    Same split as your original code:

        first 50%: training
        next 25%: validation
        final 25%: testing
    """

    train_end = round(n / 2)
    val_end = round(3 * n / 4)

    output = {
        "indices": {
            "train": np.arange(0, train_end),
            "val": np.arange(train_end, val_end),
            "test": np.arange(val_end, n),
        }
    }

    for key in ["Z_MI", "Z_MF", "Z_II", "Omega", "Y"]:
        arr = data[key]

        output[key] = {
            "train": arr[0:train_end],
            "val": arr[train_end:val_end],
            "test": arr[val_end:n],
        }

    return output


# ============================================================
# Train one PENN at one sparsity level
# ============================================================

def train_penn_one_sparsity(
    Z_train,
    Z_val,
    Z_test,
    Y_train,
    Y_val,
    Y_test,
    Omega_train,
    Omega_val,
    Omega_test,
    prune_amount,
    lr=0.001,
    epochs=200,
    weight_decay=1e-3,
    prune_start=10,
    patience=10,
    batch_size=200,
):
    model = PENN().to(device)

    train_data = TensorDataset(Z_train, Omega_train, Y_train)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    loss_fn = nn.MSELoss()

    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=0.001,
        restore_best_weights=True,
    )

    Z_val_device = Z_val.to(device)
    Z_test_device = Z_test.to(device)

    Omega_val_device = Omega_val.to(device)
    Omega_test_device = Omega_test.to(device)

    Y_val_device = Y_val.to(device)
    Y_test_device = Y_test.to(device)

    for epoch in range(epochs):
        model.train()

        if epoch == prune_start:
            model = prune_and_reinit(model, amount=prune_amount)

        for z_batch, omega_batch, y_batch in train_loader:
            z_batch = z_batch.to(device)
            omega_batch = omega_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            pred = model(z_batch, omega_batch)
            loss = loss_fn(pred, y_batch)

            loss.backward()
            optimizer.step()

        if epoch >= 10:
            model.eval()

            with torch.no_grad():
                val_loss = loss_fn(
                    model(Z_val_device, Omega_val_device),
                    Y_val_device,
                ).item()

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                break

    early_stopping.restore_weights(model)

    model.eval()

    with torch.no_grad():
        val_loss = loss_fn(
            model(Z_val_device, Omega_val_device),
            Y_val_device,
        ).item()

        test_loss = loss_fn(
            model(Z_test_device, Omega_test_device),
            Y_test_device,
        ).item()

    return {
        "model": model,
        "val_loss": val_loss,
        "test_loss": test_loss,
    }


# ============================================================
# Select sparsity by validation loss
# ============================================================

def train_best_penn_by_validation(
    Z_train,
    Z_val,
    Z_test,
    Y_train,
    Y_val,
    Y_test,
    Omega_train,
    Omega_val,
    Omega_test,
    prune_amount_vec,
    lr=0.001,
    epochs=200,
    weight_decay=1e-3,
    prune_start=10,
    patience=10,
    batch_size=200,
):
    best = None

    for prune_amount in prune_amount_vec:
        result = train_penn_one_sparsity(
            Z_train=Z_train,
            Z_val=Z_val,
            Z_test=Z_test,
            Y_train=Y_train,
            Y_val=Y_val,
            Y_test=Y_test,
            Omega_train=Omega_train,
            Omega_val=Omega_val,
            Omega_test=Omega_test,
            prune_amount=prune_amount,
            lr=lr,
            epochs=epochs,
            weight_decay=weight_decay,
            prune_start=prune_start,
            patience=patience,
            batch_size=batch_size,
        )

        if best is None or result["val_loss"] < best["val_loss"]:
            best = {
                "model": result["model"],
                "best_prune_amount": prune_amount,
                "val_loss": result["val_loss"],
                "test_loss": result["test_loss"],
            }

    return best


# ============================================================
# Extract f2 embeddings
# ============================================================

def extract_f2_embeddings(
    model,
    Omega_train,
    Omega_val,
    Omega_test,
):
    model.eval()

    Omega_train_device = Omega_train.to(device)
    Omega_val_device = Omega_val.to(device)
    Omega_test_device = Omega_test.to(device)

    with torch.no_grad():
        emb_train = model.f2(Omega_train_device).detach().cpu().numpy()
        emb_val = model.f2(Omega_val_device).detach().cpu().numpy()
        emb_test = model.f2(Omega_test_device).detach().cpu().numpy()

    return {
        "train": emb_train,
        "val": emb_val,
        "test": emb_test,
    }


# ============================================================
# Main function
# ============================================================

def one_run_get_pattern_embeddings(
    random_state=1,
    prune_amount_vec=None,
):
    """
    One run only, no repetitions.

    For each imputation method MI, MF and II:

        1. train PENN over candidate pruning levels;
        2. select pruning level by validation loss;
        3. extract self.f2(Omega_train), self.f2(Omega_val), self.f2(Omega_test);
        4. store the embeddings.
    """

    if prune_amount_vec is None:
        prune_amount_vec = [0.9, 0.8, 0.6, 0.2]

    data = generate_and_impute_data(random_state=random_state)
    split = fixed_split(data)

    Omega_train_t = to_float_tensor(split["Omega"]["train"])
    Omega_val_t = to_float_tensor(split["Omega"]["val"])
    Omega_test_t = to_float_tensor(split["Omega"]["test"])

    Y_train_t = to_float_tensor(split["Y"]["train"], response=True)
    Y_val_t = to_float_tensor(split["Y"]["val"], response=True)
    Y_test_t = to_float_tensor(split["Y"]["test"], response=True)

    Z_tensors = {
        "MI": {
            "train": to_float_tensor(split["Z_MI"]["train"]),
            "val": to_float_tensor(split["Z_MI"]["val"]),
            "test": to_float_tensor(split["Z_MI"]["test"]),
        },
        "MF": {
            "train": to_float_tensor(split["Z_MF"]["train"]),
            "val": to_float_tensor(split["Z_MF"]["val"]),
            "test": to_float_tensor(split["Z_MF"]["test"]),
        },
        "II": {
            "train": to_float_tensor(split["Z_II"]["train"]),
            "val": to_float_tensor(split["Z_II"]["val"]),
            "test": to_float_tensor(split["Z_II"]["test"]),
        },
    }

    output = {
        "metadata": {
            "random_state": random_state,
            "n": n,
            "d": d,
            "sigma": sigma,
            "width_1": width_1,
            "width_2": width_2,
            "embedding_dim": embedding_dim,
            "prune_amount_vec": prune_amount_vec,
            "train_indices": split["indices"]["train"],
            "val_indices": split["indices"]["val"],
            "test_indices": split["indices"]["test"],
            "model": "model_3",
        },
        
        "Omega": {
        "train": np.asarray(split["Omega"]["train"], dtype=np.float32),
        "val": np.asarray(split["Omega"]["val"], dtype=np.float32),
        "test": np.asarray(split["Omega"]["test"], dtype=np.float32),
        }
    }

    for method in ["MI", "MF", "II"]:
        print("\n========================================")
        print(f"Training PENN for {method}")
        print("========================================")

        best = train_best_penn_by_validation(
            Z_train=Z_tensors[method]["train"],
            Z_val=Z_tensors[method]["val"],
            Z_test=Z_tensors[method]["test"],
            Y_train=Y_train_t,
            Y_val=Y_val_t,
            Y_test=Y_test_t,
            Omega_train=Omega_train_t,
            Omega_val=Omega_val_t,
            Omega_test=Omega_test_t,
            prune_amount_vec=prune_amount_vec,
            lr=0.001,
            epochs=200,
            weight_decay=1e-3,
            prune_start=10,
            patience=10,
            batch_size=200,
        )

        embeddings = extract_f2_embeddings(
            model=best["model"],
            Omega_train=Omega_train_t,
            Omega_val=Omega_val_t,
            Omega_test=Omega_test_t,
        )

        output[method] = {
            "best_prune_amount": best["best_prune_amount"],
            "val_loss": best["val_loss"],
            "test_loss": best["test_loss"],
            "f2_embeddings": embeddings,
            "f2_state_dict": module_state_dict_cpu(best["model"].f2),
        }

        print(f"{method} finished.")
        print(f"Best pruning amount: {best['best_prune_amount']}")
        print(f"Validation loss: {best['val_loss']:.6f}")
        print(f"Test loss: {best['test_loss']:.6f}")
        print("Embedding shapes:")
        print("  train:", embeddings["train"].shape)
        print("  val:  ", embeddings["val"].shape)
        print("  test: ", embeddings["test"].shape)

    return output
