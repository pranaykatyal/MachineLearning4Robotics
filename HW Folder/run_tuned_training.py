import importlib.util
import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

HW1_PATH = os.path.join(os.path.dirname(__file__), 'HW1.py')
spec = importlib.util.spec_from_file_location('hw1mod', HW1_PATH)
hw1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hw1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run():
    # small dataset
    num_sequences = 8
    seq_len = 4
    F1, F2, F3, Alpha2, Alpha3, Tau = hw1.GenerateLargeDataset(num_sequences=num_sequences, sequence_length=seq_len)
    thruster_commands = torch.stack([F1, F2, Alpha2, F3, Alpha3], dim=1)

    inputs = Tau.numpy(); outputs = thruster_commands.numpy()
    input_scaler = StandardScaler(); output_scaler = StandardScaler()
    inputs_scaled = input_scaler.fit_transform(inputs)
    outputs_scaled = output_scaler.fit_transform(outputs)

    X = torch.tensor(inputs_scaled, dtype=torch.float32).to(DEVICE)
    Y = torch.tensor(outputs_scaled, dtype=torch.float32).to(DEVICE)

    ds = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)

    # Increase capacity and train with curriculum + augmentation
    model = hw1.Autoencoder(hidden_size=128).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)

    # Curriculum settings (anneal zero_reg down, increase k0)
    zero_start = 200.0
    zero_end = 5.0
    k0_start = 0.1
    k0_end = 2.0

    zero_phys = np.zeros((1,3), dtype=np.float32)
    zero_scaled = input_scaler.transform(zero_phys)
    zero_scaled_t = torch.tensor(zero_scaled, dtype=torch.float32).to(DEVICE)

    epochs = 200
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=60, gamma=0.5)
    max_zero_per_batch = 3
    clip_grad = 1.0
    for ep in range(epochs):
        # linear anneal
        alpha = ep / float(max(1, epochs-1))
        zero_weight = zero_start * (1 - alpha) + zero_end * alpha
        k0 = k0_start * (1 - alpha) + k0_end * alpha

        total=0.0
        model.train()
        for xb,yb in loader:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE)
            # batch zero augmentation: append a few zero samples per batch
            n_zero = min(max_zero_per_batch, xb.shape[0])
            if n_zero > 0:
                z_x = zero_scaled_t.repeat(n_zero, 1)
                z_y = torch.zeros((n_zero, yb.shape[1]), dtype=yb.dtype, device=DEVICE)
                xb = torch.cat([xb, z_x], dim=0)
                yb = torch.cat([yb, z_y], dim=0)

            opt.zero_grad()
            enc, dec = model(xb)
            L1 = hw1.ComputeL1Loss(dec, xb)
            L0 = hw1.ComputeL0Loss(enc, xb, output_scaler, input_scaler, DEVICE)
            _, decz = model(zero_scaled_t)
            Lz = hw1.ComputeL1Loss(decz, zero_scaled_t)
            loss = k0*L0 + 1.0*L1 + zero_weight * Lz
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            total += loss.item()
        scheduler.step()
        if (ep+1) % 10 == 0:
            print(f'Epoch {ep+1}/{epochs} lr={scheduler.get_last_lr()[0]:.2e} k0={k0:.3f} z={zero_weight:.1f} loss {total:.6f}')

    model.eval()
    with torch.no_grad():
        _, decz = model(zero_scaled_t)
        dec_np = decz.cpu().numpy().reshape(-1)
        phys = dec_np * input_scaler.scale_ + input_scaler.mean_
        print('Final decoded(0) phys:', phys, 'L2:', np.linalg.norm(phys))
    # Save model and scalers for test_zero to pick up
    save_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'autoencoder.pth'))
    with open(os.path.join(save_dir, 'scalers.pkl'), 'wb') as f:
        import pickle
        pickle.dump({'input_scaler': input_scaler, 'output_scaler': output_scaler}, f)
    print('Saved model and scalers to', save_dir)

if __name__=='__main__':
    run()
