import importlib.util
import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Load HW1 as module
HW1_PATH = os.path.join(os.path.dirname(__file__), 'HW1.py')
spec = importlib.util.spec_from_file_location('hw1mod', HW1_PATH)
hw1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hw1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_trial(z_weight, k0_weight, epochs=12):
    # small dataset
    num_sequences = 8
    seq_len = 4
    F1, F2, F3, Alpha2, Alpha3, Tau = hw1.GenerateLargeDataset(num_sequences=num_sequences, sequence_length=seq_len)
    thruster_commands = torch.stack([F1, F2, Alpha2, F3, Alpha3], dim=1)

    inputs = Tau.numpy()
    outputs = thruster_commands.numpy()
    input_scaler = StandardScaler(); output_scaler = StandardScaler()
    inputs_scaled = input_scaler.fit_transform(inputs)
    outputs_scaled = output_scaler.fit_transform(outputs)

    X = torch.tensor(inputs_scaled, dtype=torch.float32).to(DEVICE)
    Y = torch.tensor(outputs_scaled, dtype=torch.float32).to(DEVICE)

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    model = hw1.Autoencoder(hidden_size=64).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # set weights locally
    k0 = k0_weight
    # other ks same as default
    k1 = 1; k2=0.1; k3=1e-7; k4=1e-7; k5=0.1

    # precompute zero
    zero_phys = np.zeros((1,3), dtype=np.float32)
    zero_scaled = input_scaler.transform(zero_phys)
    zero_scaled_t = torch.tensor(zero_scaled, dtype=torch.float32).to(DEVICE)

    for epoch in range(epochs):
        model.train()
        for xb,yb in loader:
            opt.zero_grad()
            encoded, decoded = model(xb)
            L1 = hw1.ComputeL1Loss(decoded, xb)
            L0 = hw1.ComputeL0Loss(encoded, xb, output_scaler, input_scaler, DEVICE)
            # zero reg
            encoded_z, decoded_z = model(zero_scaled_t)
            Lz = hw1.ComputeL1Loss(decoded_z, zero_scaled_t)
            loss = k0*L0 + k1*L1 + z_weight * Lz
            loss.backward(); opt.step()

    # evaluate decoded zero in physical units
    model.eval()
    with torch.no_grad():
        _, decoded_z = model(zero_scaled_t)
        decoded_np = decoded_z.cpu().numpy().reshape(-1)
        # unscale using input_scaler (scaled -> phys)
        phys = decoded_np * input_scaler.scale_ + input_scaler.mean_
        l2 = np.linalg.norm(phys)
    return l2, phys

if __name__ == '__main__':
    z_weights = [0.0, 1.0, 10.0, 50.0, 100.0]
    k0_weights = [0.0, 0.5, 1.0, 2.0]

    results = []
    for z in z_weights:
        for k0 in k0_weights:
            print(f'Trial z={z}, k0={k0}')
            l2, phys = run_trial(z, k0, epochs=18)
            print('  decoded phys:', phys, 'L2:', l2)
            results.append((z,k0,l2,phys))

    # find best (min L2)
    best = min(results, key=lambda x: x[2])
    print('\nBest config: zero_weight={}, k0={} -> L2={} decoded={}'.format(best[0], best[1], best[2], best[3]))
