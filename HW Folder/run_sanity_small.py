import importlib.util
import os
import sys
from pathlib import Path

# Path to the HW1.py module (contains model and helper functions)
HW1_PATH = os.path.join(os.path.dirname(__file__), 'HW1.py')

spec = importlib.util.spec_from_file_location('hw1mod', HW1_PATH)
hw1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hw1)

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

print('Loaded HW1 module from', HW1_PATH)

def run_small_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Generate a very small dataset
    num_sequences = 8
    seq_len = 4
    F1, F2, F3, Alpha2, Alpha3, Tau = hw1.GenerateLargeDataset(num_sequences=num_sequences, sequence_length=seq_len)
    # Depending on how GenerateLargeDataset is implemented it may return concatenated tensors.
    # Build thruster commands as in the main script
    thruster_commands = torch.stack([F1, F2, Alpha2, F3, Alpha3], dim=1)  # [N,5]

    inputs = Tau.numpy()    # [N,3]
    outputs = thruster_commands.numpy()  # [N,5]

    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    inputs_scaled = input_scaler.fit_transform(inputs)
    outputs_scaled = output_scaler.fit_transform(outputs)

    X = torch.tensor(inputs_scaled, dtype=torch.float32)
    Y = torch.tensor(outputs_scaled, dtype=torch.float32)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = hw1.Autoencoder(hidden_size=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 8
    # Precompute scaled zero for regularization (shape [1, features])
    zero_phys = np.zeros((1, 3), dtype=np.float32)
    zero_scaled = input_scaler.transform(zero_phys)  # [1,3]
    zero_scaled_t = torch.tensor(zero_scaled, dtype=torch.float32).to(device)

    zero_reg_weight = 10.0  # weight for zero-point regularizer
    for epoch in range(num_epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            encoded, decoded = model(xb)
            # Use simple reconstruction loss (decoded tau vs xb)
            L1 = hw1.ComputeL1Loss(decoded, xb)
            # Also include L0 physics loss using output_scaler
            L0 = hw1.ComputeL0Loss(encoded, xb, output_scaler, input_scaler, device)
            # Zero-input regularizer: encourage decoded(zero) == zero (in scaled space)
            encoded_z, decoded_z = model(zero_scaled_t)
            Lz = hw1.ComputeL1Loss(decoded_z, zero_scaled_t)
            loss = L0 + L1 + zero_reg_weight * Lz
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs} loss: {total:.6f}')

    print('Training done. Running sanity test:')
    try:
        hw1.sanity_zero_test(model, input_scaler, output_scaler, device)
    except Exception as e:
        print('Sanity test raised exception:', e)

if __name__ == '__main__':
    run_small_training()
