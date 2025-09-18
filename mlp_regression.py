from sklearn.datasets import fetch_california_housing
# info about the dataset:
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html


import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
data = fetch_california_housing()
print(data.feature_names)

X, y = data.data, data.target

# Define the model
model = nn.Sequential(
    nn.Linear(8, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
).to(device)  # Move model to device

# Loss function and optimizer
loss_fn = nn.MSELoss()  # Mean Square Error
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# Train-test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)  # Move data to device
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)  # Move data to device
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)  # Move data to device
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)  # Move data to device

# Training parameters
n_epochs = 100  # Number of epochs to run
batch_size = 100  # Size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf  # Initialize to infinity
best_weights = None
history_eval = []
history_train = []

# Training loop
for epoch in range(n_epochs):
    model.train()
    print(f"Epoch {epoch+1}/{n_epochs}")
    for start in batch_start:
        # Take a batch
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        
        # Forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        
        history_train.append(loss.item())
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update weights
        optimizer.step()
        

        # Print progress for each batch
        print(f" epoch: {epoch+1}/{n_epochs},  Batch {start//batch_size+1}/{len(batch_start)}, Loss: {loss.item():.6f}")
    
    # Evaluate accuracy at the end of each epoch
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test).item()
        history_eval.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

# Restore model and return best accuracy
model.load_state_dict(best_weights)

print(f"Best MSE: {best_mse:.2f}")
print(f"Best RMSE: {np.sqrt(best_mse):.2f}")
plt.figure(1)
plt.plot(history_eval)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE vs Epochs in eval")

plt.figure(2)
plt.plot(history_train)
plt.title("MSE in train")


plt.show()
