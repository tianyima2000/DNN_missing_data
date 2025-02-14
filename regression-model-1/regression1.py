def train_test_model(model, Z, Y, lr, l1_lambda, Omega=None, epochs=200, patience=10, threshold=0.001, live_plot=False):
    # model: the neural network to train
    # Z: covariate matrix (n times d), including training, validation and testing data
    # Y: response vector (n-dimensional), including training, validation and testing data
    # lr: learning rate for Adam
    # lambda: regularisation parameter (l1 penalty)
    # Omega: If Omega=None, then a classical neural network is used. 
    #        If Omega is the observation pattern matrix (n times d), including training, validation and testing data, then PENN is used.
    # epochs: maximum number of epochs to train
    # patience: patience for early stopping
    # threshold: after training, set every parameter below the threshold to zero
    # live_plot: if True, then it will plot the training loss and validation loss live

    
    ##### if Omega is not None
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
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        train_losses = []
        val_losses = []

        ### start training
        for epoch in range(epochs):

            model.train()
            for z_batch, omega_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(z_batch, omega_batch)
                loss = loss_fn(pred, y_batch)

                # L1 penalty
                l1_penalty = 0
                for param in model.parameters():
                    l1_penalty += torch.sum(torch.abs(param))
                loss += l1_lambda * l1_penalty

                loss.backward()
                optimizer.step()
                
            if epoch >= 4:
                model.eval()
                with torch.no_grad():
                    train_losses.append(loss_fn(model(Z_train, Omega_train), Y_train))
                    val_losses.append(loss_fn(model(Z_val, Omega_val), Y_val))

                # Live plotting
                if live_plot:
                    clear_output(wait=True)  # Clear previous output (Jupyter only)
                    plt.figure(figsize=(8, 5))
                    plt.plot(range(5,epoch+2), train_losses, label='Training Loss')
                    plt.plot(range(5,epoch+2), val_losses, label='Validation Loss')
                    plt.title(f'Epoch {epoch+1}/{epochs}')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.show()

                # Early stopping
                min_index = val_losses.index(min(val_losses))
                if epoch - min_index >= patience:
                    break
                
        ### set every parameter below threshold to zero
        with torch.no_grad():
            for param in model.parameters():
                param.data[param.abs() <= threshold] = 0     
        
        ### return testing loss
        return loss_fn(model(Z_test, Omega_test), Y_test)


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
        train_data = TensorDataset(Z_train, Omega_train, Y_train)
        train_loader = DataLoader(dataset = train_data, batch_size=200, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        train_losses = []
        val_losses = []

        ### start training
        for epoch in range(epochs):

            model.train()
            for z_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(z_batch)
                loss = loss_fn(pred, y_batch)

                # L1 penalty
                l1_penalty = 0
                for param in model.parameters():
                    l1_penalty += torch.sum(torch.abs(param))
                loss += l1_lambda * l1_penalty

                loss.backward()
                optimizer.step()
                
            if epoch >= 4:
                model.eval()
                with torch.no_grad():
                    train_losses.append(loss_fn(model(Z_train), Y_train))
                    val_losses.append(loss_fn(model(Z_val), Y_val))

                # Live plotting
                if live_plot:
                    clear_output(wait=True)  # Clear previous output (Jupyter only)
                    plt.figure(figsize=(8, 5))
                    plt.plot(range(5,epoch+2), train_losses, label='Training Loss')
                    plt.plot(range(5,epoch+2), val_losses, label='Validation Loss')
                    plt.title(f'Epoch {epoch+1}/{epochs}')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.show()

                # Early stopping
                min_index = val_losses.index(min(val_losses))
                if epoch - min_index >= patience:
                    break
                
        ### set every parameter below threshold to zero
        with torch.no_grad():
            for param in model.parameters():
                param.data[param.abs() <= threshold] = 0     
        
        ### return testing loss
        return loss_fn(model(Z_test), Y_test)