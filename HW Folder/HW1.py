# Authors : Pranay Katyal, Anirudh Ramanathan
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pickle
import os
import sys
import shutil
from datetime import datetime


def clear_tensorboard_logs():
    log_dir = "runs/control_allocation_experiment"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    return log_dir


def RandomWalk(Var, Step, minVal, maxVal):
    Var += np.random.uniform(-Step, Step)
    Var = np.clip(Var, minVal, maxVal)
    return Var


def GenerateData(n=100):
    """Generate data with PHYSICAL UNITS"""
    F1 = np.zeros(n)
    F1[0] = np.random.uniform(-8000, 8000)
    for i in range(1, n):
        F1[i] = RandomWalk(F1[i-1], 800, -12000, 12000)
    F2 = np.zeros(n)
    F3 = np.zeros(n)
    F2[0] = np.random.uniform(-4000, 4000)
    F3[0] = np.random.uniform(-4000, 4000)
    for i in range(1, n):
        F2[i] = RandomWalk(F2[i-1], 400, -6000, 6000)
        F3[i] = RandomWalk(F3[i-1], 400, -6000, 6000)
    Alpha2 = np.zeros(n)
    Alpha3 = np.zeros(n)
    Alpha2[0] = np.random.uniform(-150, 150)
    Alpha3[0] = np.random.uniform(-150, 150)
    for i in range(1, n):
        Alpha2[i] = RandomWalk(Alpha2[i-1], 8, -200, 200)
        Alpha3[i] = RandomWalk(Alpha3[i-1], 8, -200, 200)
    F1 = torch.tensor(F1, dtype=torch.float32)
    F2 = torch.tensor(F2, dtype=torch.float32)
    F3 = torch.tensor(F3, dtype=torch.float32)
    Alpha2 = torch.tensor(Alpha2, dtype=torch.float32)
    Alpha3 = torch.tensor(Alpha3, dtype=torch.float32)
    return F1, F2, F3, Alpha2, Alpha3


def GenerateBmatrix(Alpha2, Alpha3, n=100):
    """Generate B matrix using PHYSICAL angles in degrees"""
    B = torch.zeros((n, 3, 3))
    l1, l2, l3, l4 = -14, 14.5, -2.7, 2.7
    a2_rad = Alpha2 * torch.pi / 180
    a3_rad = Alpha3 * torch.pi / 180
    B[:, 0, 1] = torch.cos(a2_rad)
    B[:, 0, 2] = torch.cos(a3_rad)
    B[:, 1, 0] = 1
    B[:, 1, 1] = torch.sin(a2_rad)
    B[:, 1, 2] = torch.sin(a3_rad)
    B[:, 2, 0] = l2
    B[:, 2, 1] = l1*torch.sin(a2_rad) - l3*torch.cos(a2_rad)
    B[:, 2, 2] = l1*torch.sin(a3_rad) - l4*torch.cos(a3_rad)
    return B


def ComputeTau(F1, F2, F3, B):
    """Compute tau using PHYSICAL forces in Newtons"""
    F = torch.stack((F1, F2, F3), dim=1)
    Tau = torch.bmm(B, F.unsqueeze(-1)).squeeze(-1)
    return Tau


def print_progress_bar(current, total, bar_length=50):
    percentage = current / total
    filled_length = int(bar_length * percentage)
    bar = '#' * filled_length + '.' * (bar_length - filled_length)
    sys.stdout.write(f'\rProgress: [{bar}] {percentage*100:.1f}% ({current}/{total})')
    sys.stdout.flush()


def GenerateLargeDataset(num_sequences=800, sequence_length=100):
    print(f"Generating {num_sequences * sequence_length:,} samples...")
    all_F1, all_F2, all_F3, all_Alpha2, all_Alpha3, all_Tau = [], [], [], [], [], []
    for seq in range(num_sequences):
        F1, F2, F3, Alpha2, Alpha3 = GenerateData(sequence_length)
        B = GenerateBmatrix(Alpha2, Alpha3, sequence_length)
        Tau = ComputeTau(F1, F2, F3, B)
        all_F1.append(F1)
        all_F2.append(F2)
        all_F3.append(F3)
        all_Alpha2.append(Alpha2)
        all_Alpha3.append(Alpha3)
        all_Tau.append(Tau)
        if (seq + 1) % 50 == 0 or seq == num_sequences - 1:
            print_progress_bar(seq + 1, num_sequences)
    print("\nConcatenating sequences...")
    F1_final = torch.cat(all_F1)
    F2_final = torch.cat(all_F2)
    F3_final = torch.cat(all_F3)
    Alpha2_final = torch.cat(all_Alpha2)
    Alpha3_final = torch.cat(all_Alpha3)
    Tau_final = torch.cat(all_Tau)
    print(f"Dataset generated: {len(F1_final):,} samples")
    return F1_final, F2_final, F3_final, Alpha2_final, Alpha3_final, Tau_final


class Encoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=5, dropout_rate=0.2):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.init_weights()
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name or 'weight_hh' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        last_out = out[:, -1, :]
        x = self.fc1(last_out)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output    

class Decoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=3, dropout_rate=0.2):
        super(Decoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.init_weights()
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name or 'weight_hh' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        last_out = out[:, -1, :]
        x = self.fc1(last_out)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output 

class Autoencoder(nn.Module):
    def __init__(self, hidden_size=64, dropout_rate=0.2):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size=3, hidden_size=hidden_size, output_size=5, dropout_rate=dropout_rate)
        self.decoder = Decoder(input_size=5, hidden_size=hidden_size, output_size=3, dropout_rate=dropout_rate)
        self.register_buffer('max_limits', torch.tensor([10000.0, 5000.0, 180.0, 5000.0, 180.0]))
        self.register_buffer('max_rates', torch.tensor([1000.0, 250.0, 10.0, 250.0, 10.0]))
        self.register_buffer('command_scales', torch.tensor([10000.0, 5000.0, 180.0, 5000.0, 180.0]))
        self.register_buffer('tau_scale', torch.tensor([100000.0, 100000.0, 1000000.0]))
    def forward(self, x):
        x_normalized = x / self.tau_scale
        encoded_raw = self.encoder(x_normalized)
        encoded = torch.tanh(encoded_raw) * self.max_limits
        encoded_normalized = encoded / self.command_scales
        decoded_raw = self.decoder(encoded_normalized)
        decoded = decoded_raw * self.tau_scale
        return encoded, decoded


def ComputeL0Loss(encoded_commands, original_tau, device):
    F1 = encoded_commands[:, 0]
    F2 = encoded_commands[:, 1]
    Alpha2 = encoded_commands[:, 2]
    F3 = encoded_commands[:, 3]
    Alpha3 = encoded_commands[:, 4]
    n = F1.shape[0]
    B = GenerateBmatrix(Alpha2, Alpha3, n).to(device)
    reconstructed_tau = ComputeTau(F1, F2, F3, B)
    tau_scale = torch.tensor([100000.0, 100000.0, 1000000.0]).to(device)
    reconstructed_normalized = reconstructed_tau / tau_scale
    original_normalized = original_tau / tau_scale
    return nn.MSELoss()(reconstructed_normalized, original_normalized)


def ComputeL1Loss(decoded_tau, original_tau):
    tau_scale = torch.tensor([100000.0, 100000.0, 1000000.0]).to(decoded_tau.device)
    decoded_normalized = decoded_tau / tau_scale
    original_normalized = original_tau / tau_scale
    return nn.MSELoss()(decoded_normalized, original_normalized)


def ComputeL2Loss(encoded_commands, max_limits, device):
    violations = torch.relu(torch.abs(encoded_commands) - max_limits.to(device))
    return torch.mean(violations ** 2)


def ComputeL3Loss(encoded_commands, max_rates, device):
    if encoded_commands.size(0) > 1:
        rate_changes = torch.abs(encoded_commands[1:] - encoded_commands[:-1])
        rate_scale = torch.tensor([10000.0, 5000.0, 180.0, 5000.0, 180.0]).to(device)
        rate_changes_norm = rate_changes / rate_scale
        max_rates_norm = max_rates / rate_scale
        violations = torch.relu(rate_changes_norm - max_rates_norm.to(device))
        return torch.mean(violations ** 2)
    return torch.tensor(0.0).to(device)


def ComputeL4Loss(encoded_commands):
    F1_norm = encoded_commands[:, 0] / 10000.0
    F2_norm = encoded_commands[:, 1] / 5000.0   
    F3_norm = encoded_commands[:, 3] / 5000.0
    power = F1_norm**2 + F2_norm**2 + F3_norm**2
    return power.mean()


def ComputeL5Loss(encoded_commands, device):
    Alpha2 = encoded_commands[:, 2]
    Alpha3 = encoded_commands[:, 4]
    penalty = 0
    for alpha in [Alpha2, Alpha3]:
        penalty += torch.sum(torch.relu(1 - torch.abs(alpha + 90) / 10) ** 2)
        penalty += torch.sum(torch.relu(1 - torch.abs(alpha - 90) / 10) ** 2)
    return penalty / len(encoded_commands)


def evaluate_model(model, data_loader, device, mode="Validation"):
    model.eval()
    total_loss = 0
    total_l0, total_l1, total_l2, total_l3, total_l4, total_l5 = 0, 0, 0, 0, 0, 0
    num_batches = 0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            encoded, decoded = model(batch_x)
            L0 = ComputeL0Loss(encoded, batch_x, device)
            L1 = ComputeL1Loss(decoded, batch_x)
            L2 = ComputeL2Loss(encoded, model.max_limits, device)
            L3 = ComputeL3Loss(encoded, model.max_rates, device)
            L4 = ComputeL4Loss(encoded)
            L5 = ComputeL5Loss(encoded, device)
            Loss = 10.0*L0 + 1.0*L1 + 0.1*L2 + 0.01*L3 + 0.001*L4 + 0.1*L5
            total_loss += Loss.item()
            total_l0 += L0.item()
            total_l1 += L1.item()
            total_l2 += L2.item()
            total_l3 += L3.item()
            total_l4 += L4.item()
            total_l5 += L5.item()
            num_batches += 1
    results = {
        'total_loss': total_loss / num_batches,
        'L0': total_l0 / num_batches,
        'L1': total_l1 / num_batches,
        'L2': total_l2 / num_batches,
        'L3': total_l3 / num_batches,
        'L4': total_l4 / num_batches,
        'L5': total_l5 / num_batches
    }
    print(f"{mode} Results:")
    print(f"  Total Loss: {results['total_loss']:.6f}")
    print(f"  L0 (Physics): {results['L0']:.6f}")
    print(f"  L1 (Autoencoder): {results['L1']:.6f}")
    print(f"  L2 (Magnitude): {results['L2']:.6f}")
    print(f"  L3 (Rate): {results['L3']:.6f}")
    print(f"  L4 (Power): {results['L4']:.6f}")
    print(f"  L5 (Sector): {results['L5']:.6f}")
    return results


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    log_dir = clear_tensorboard_logs()
    print("Generating training data...")
    F1, F2, F3, Alpha2, Alpha3, Tau = GenerateLargeDataset(num_sequences=10000, sequence_length=100)
    thruster_commands = torch.stack([F1, F2, Alpha2, F3, Alpha3], dim=1)
    print("\nDebug: Checking first data point:")
    first_tau = Tau[0]
    first_commands = thruster_commands[0]
    B_first = GenerateBmatrix(torch.tensor([first_commands[2]]), torch.tensor([first_commands[4]]), n=1)
    tau_physics = ComputeTau(
        torch.tensor([first_commands[0]]), 
        torch.tensor([first_commands[1]]), 
        torch.tensor([first_commands[3]]), 
        B_first
    )[0]
    print(f"Input tau:     [{first_tau[0]:10.2f}, {first_tau[1]:10.2f}, {first_tau[2]:10.2f}]")
    print(f"Physics tau:   [{tau_physics[0]:10.2f}, {tau_physics[1]:10.2f}, {tau_physics[2]:10.2f}]")
    print(f"Commands: F1={first_commands[0]:8.2f}N, F2={first_commands[1]:8.2f}N, α2={first_commands[2]:6.1f}°")
    print(f"          F3={first_commands[3]:8.2f}N, α3={first_commands[4]:6.1f}°")
    error = torch.norm(first_tau - tau_physics) / torch.norm(first_tau)
    print(f"Relative Error: {error.item():.2%}\n")
    inputs_tensor = Tau
    outputs_tensor = thruster_commands
    total_samples = len(inputs_tensor)
    train_end = int(0.7 * total_samples)
    val_end = int(0.85 * total_samples)
    X_train = inputs_tensor[:train_end]
    X_val = inputs_tensor[train_end:val_end] 
    X_test = inputs_tensor[val_end:]
    y_train = outputs_tensor[:train_end]
    y_val = outputs_tensor[train_end:val_end]
    y_test = outputs_tensor[val_end:]
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(hidden_size=32, dropout_rate=0.2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.01,
        epochs=100,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"{log_dir}/fixed_{timestamp}")
    print("\nTesting model architecture:")
    test_input = X_train[:5].to(device)
    with torch.no_grad():
        encoded, decoded = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Encoded shape: {encoded.shape}")
        print(f"Decoded shape: {decoded.shape}")
        print(f"Encoded values range: [{encoded.min().item():.1f}, {encoded.max().item():.1f}]")
    num_epochs = 100
    best_val_loss = float('inf')
    best_epoch = 0
    step = 1
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            encoded, decoded = model(batch_x)
            L0 = ComputeL0Loss(encoded, batch_x, device)
            L1 = ComputeL1Loss(decoded, batch_x)
            L2 = ComputeL2Loss(encoded, model.max_limits, device)
            L3 = ComputeL3Loss(encoded, model.max_rates, device)
            L4 = ComputeL4Loss(encoded)
            L5 = ComputeL5Loss(encoded, device)
            Loss = 10.0*L0 + 1.0*L1 + 0.1*L2 + 0.01*L3 + 0.001*L4 + 0.1*L5
            Loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            if step % 20 == 0:
                writer.add_scalar('Loss/L0_Physics', L0.item(), step)
                writer.add_scalar('Loss/L1_Autoencoder', L1.item(), step)
                writer.add_scalar('Loss/L2_Magnitude', L2.item(), step)
                writer.add_scalar('Loss/L3_Rate', L3.item(), step)
                writer.add_scalar('Loss/L4_Power', L4.item(), step)
                writer.add_scalar('Loss/L5_Sector', L5.item(), step)
                writer.add_scalar('Loss/Total', Loss.item(), step)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], step)
            total_loss += Loss.item()
            num_batches += 1
            step += 1
        avg_train_loss = total_loss / num_batches
        if epoch % 5 == 0:
            val_results = evaluate_model(model, val_loader, device, "Validation")
            if val_results['total_loss'] < best_val_loss:
                best_val_loss = val_results['total_loss']
                best_epoch = epoch
                torch.save(model.state_dict(), 'best_model_fixed.pth')
            writer.add_scalar('Epoch/Train_Loss', avg_train_loss, epoch)
            writer.add_scalar('Epoch/Val_Loss', val_results['total_loss'], epoch)
    writer.close()
    model.load_state_dict(torch.load('best_model_fixed.pth'))
    test_results = evaluate_model(model, test_loader, device, "Test")
    model.eval()
    with torch.no_grad():
        physical_tau_requests = torch.tensor([
            [10000, 0, 0],
            [0, 5000, 0],
            [0, 0, 50000],
            [5000, 2500, 25000],
            [-8000, -3000, -30000],
        ], dtype=torch.float32).to(device)
        encoded, decoded = model(physical_tau_requests)
        for i in range(len(physical_tau_requests)):
            tau_req = physical_tau_requests[i].cpu().numpy()
            commands = encoded[i].cpu().numpy()
            B_single = GenerateBmatrix(
                torch.tensor([commands[2]], dtype=torch.float32), 
                torch.tensor([commands[4]], dtype=torch.float32), 
                1)
            tau_physics = ComputeTau(
                torch.tensor([commands[0]], dtype=torch.float32), 
                torch.tensor([commands[1]], dtype=torch.float32), 
                torch.tensor([commands[3]], dtype=torch.float32), 
                B_single
            ).numpy()[0]
            physics_error = np.linalg.norm(tau_req - tau_physics) / np.linalg.norm(tau_req)
    torch.save(model.state_dict(), 'ship_control_allocator.pth')