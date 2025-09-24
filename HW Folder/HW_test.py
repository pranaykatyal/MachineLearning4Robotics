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


def RandomWalk(Var, Step, minVal, maxVal):
    Var += np.random.uniform(-Step, Step)
    Var = np.clip(Var, minVal, maxVal)
    return Var

def GenerateData(n = 100):
    # Front Thrusters
    F1 = np.zeros(n)
    F1[0] = np.random.uniform(-10000, 10000)
    for i  in range(1, n):
        F1[i] = RandomWalk(F1[i-1],500,-10000,10000)
    
    # Rear Azimuth Thrusters
    F2 = np.zeros(n)
    F3 = np.zeros(n)
    F2[0] = np.random.uniform(-5000, 5000)
    F3[0] = np.random.uniform(-5000, 5000)
    
    for i in range(1, n):
        F2[i] = RandomWalk(F2[i-1], 250, -5000, 5000)
        F3[i] = RandomWalk(F3[i-1], 250, -5000, 5000)
    
    # Azimuth Angles
    Alpha2 = np.zeros(n)
    Alpha3 = np.zeros(n)
    Alpha2[0] = np.random.uniform(-180, 180)
    Alpha3[0] = np.random.uniform(-180, 180)
    
    for i in range(1, n):
        Alpha2[i] = RandomWalk(Alpha2[i-1], 6, -180, 180)
        Alpha3[i] = RandomWalk(Alpha3[i-1], 6, -180, 180)
    
    # Converting to tensors
    F1 = torch.tensor(F1, dtype=torch.float32)
    F2 = torch.tensor(F2, dtype=torch.float32)
    F3 = torch.tensor(F3, dtype=torch.float32)
    Alpha2 = torch.tensor(Alpha2, dtype=torch.float32)
    Alpha3 = torch.tensor(Alpha3, dtype=torch.float32)
    
    return F1, F2, F3, Alpha2, Alpha3

def GenerateBmatrix(Alpha2, Alpha3, n = 100):
    B = torch.zeros((n, 3, 3))
    # Lengths
    l1,l2,l3,l4 = 14.5, 14, 2.7, 2.7
    
    # Convert degrees to radians
    a2_rad = Alpha2* torch.pi / 180
    a3_rad = Alpha3 * torch.pi / 180
    
    # Making the B Matrix
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
    F = torch.stack((F1, F2, F3), dim=1)
    Tau = torch.bmm(B, F.unsqueeze(-1)).squeeze(-1)
    return Tau 

def print_progress_bar(current, total, bar_length=50):
    percentage = current / total
    filled_length = int(bar_length * percentage)
    bar = '#' * filled_length + '.' * (bar_length - filled_length)
    sys.stdout.write(f'\rProgress: [{bar}] {percentage*100:.1f}% ({current}/{total})')
    sys.stdout.flush()

def GenerateLargeDataset(num_sequences=1000, sequence_length=500):
    print(f"Generating {num_sequences * sequence_length:,} samples...")
    
    all_F1, all_F2, all_F3, all_Alpha2, all_Alpha3, all_Tau = [], [], [], [], [], []
    
    for seq in range(num_sequences):
        # Generate one sequence
        F1, F2, F3, Alpha2, Alpha3 = GenerateData(sequence_length)
        B = GenerateBmatrix(Alpha2, Alpha3, sequence_length)
        Tau = ComputeTau(F1, F2, F3, B)
        
        # Store results
        all_F1.append(F1)
        all_F2.append(F2)
        all_F3.append(F3)
        all_Alpha2.append(Alpha2)
        all_Alpha3.append(Alpha3)
        all_Tau.append(Tau)
        
        # Update progress bar every 10 sequences
        if (seq + 1) % 10 == 0 or seq == num_sequences - 1:
            print_progress_bar(seq + 1, num_sequences)
    
    print("\nConcatenating sequences...")
    
    # Concatenate all sequences
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
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        # Now we expect proper sequences, no more unsqueeze
        
        # Pass through LSTM layers
        out1, _ = self.lstm1(x)  # [batch, seq_len, hidden]
        out1 = self.dropout(out1)
        out2, _ = self.lstm2(out1)  # [batch, seq_len, hidden]
        out2 = self.dropout(out2)
        
        # Apply linear layer to ALL timesteps, not just last one
        batch_size, seq_len, hidden_size = out2.shape
        out2_reshaped = out2.reshape(-1, hidden_size)  # [batch*seq_len, hidden]
        output_flat = self.linear(out2_reshaped)  # [batch*seq_len, output_size]
        output = output_flat.reshape(batch_size, seq_len, -1)  # [batch, seq_len, output_size]
        
        return output

class Decoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=3, dropout_rate=0.2):
        super(Decoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        
        out1, _ = self.lstm1(x)
        out1 = self.dropout(out1)
        out2, _ = self.lstm2(out1)
        out2 = self.dropout(out2)
        
        # Apply linear to ALL timesteps
        batch_size, seq_len, hidden_size = out2.shape
        out2_reshaped = out2.reshape(-1, hidden_size)
        output_flat = self.linear(out2_reshaped)
        output = output_flat.reshape(batch_size, seq_len, -1)
        
        return output

class Autoencoder(nn.Module):
    def __init__(self, hidden_size=64, dropout_rate=0.2):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size=3, hidden_size=hidden_size, output_size=5, dropout_rate=dropout_rate)
        self.decoder = Decoder(input_size=5, hidden_size=hidden_size, output_size=3, dropout_rate=dropout_rate)
        
    def forward(self, x):
        # x: [batch, seq_len, 3]
        encoded = self.encoder(x)    # [batch, seq_len, 5]
        decoded = self.decoder(encoded)  # [batch, seq_len, 3]
        return encoded, decoded

def ComputeSequenceL0Loss(encoded_commands, original_tau, output_scaler, device):
    """Compute L0 loss for sequences"""
    # Flatten sequences for processing
    batch_size, seq_len, _ = encoded_commands.shape
    encoded_flat = encoded_commands.reshape(-1, 5)  # [batch*seq_len, 5]
    tau_flat = original_tau.reshape(-1, 3)          # [batch*seq_len, 3]
    
    # Extract components from encoded_commands
    F1_scaled = encoded_flat[:, 0]
    F2_scaled = encoded_flat[:, 1]
    Alpha2_scaled = encoded_flat[:, 2]
    F3_scaled = encoded_flat[:, 3]
    Alpha3_scaled = encoded_flat[:, 4]
    n = F1_scaled.shape[0]
    
    # Rescale Alpha2 and Alpha3 from scaled values back to original range
    alpha2_mean = torch.tensor(output_scaler.mean_[2], device=device)
    alpha2_std = torch.tensor(output_scaler.scale_[2], device=device)
    alpha3_mean = torch.tensor(output_scaler.mean_[4], device=device)  
    alpha3_std = torch.tensor(output_scaler.scale_[4], device=device)
    Alpha2 = Alpha2_scaled * alpha2_std + alpha2_mean
    Alpha3 = Alpha3_scaled * alpha3_std + alpha3_mean
    Alpha2 = torch.clamp(Alpha2, -180, 180)
    Alpha3 = torch.clamp(Alpha3, -180, 180)
    
    # Rescale F1,F2,F3 from scaled values back to original range
    f1_mean = torch.tensor(output_scaler.mean_[0], device=device)
    f1_std = torch.tensor(output_scaler.scale_[0], device=device)
    f2_mean = torch.tensor(output_scaler.mean_[1], device=device)
    f2_std = torch.tensor(output_scaler.scale_[1], device=device)
    f3_mean = torch.tensor(output_scaler.mean_[3], device=device)
    f3_std = torch.tensor(output_scaler.scale_[3], device=device)
    F1 = F1_scaled * f1_std + f1_mean
    F2 = F2_scaled * f2_std + f2_mean
    F3 = F3_scaled * f3_std + f3_mean
    
    # Build B matrices
    B = GenerateBmatrix(Alpha2, Alpha3, n).to(device)
    
    # Compute reconstructed tau
    reconstructed_tau = ComputeTau(F1, F2, F3, B)
    
    # Return MSE loss
    loss = nn.MSELoss()
    loss_value = loss(reconstructed_tau, tau_flat)
    
    return loss_value

def ComputeSequenceL1Loss(decoded_tau, original_tau):
    """Compute L1 loss for sequences"""
    loss = nn.MSELoss()
    return loss(decoded_tau, original_tau)

def ComputeSequenceL2Loss(encoded_commands, output_scaler, device):
    """Compute L2 loss for sequences"""
    batch_size, seq_len, _ = encoded_commands.shape
    encoded_flat = encoded_commands.reshape(-1, 5)
    
    # Extract and unscale components
    F1_scaled = encoded_flat[:, 0]
    F2_scaled = encoded_flat[:, 1]
    Alpha2_scaled = encoded_flat[:, 2]
    F3_scaled = encoded_flat[:, 3]
    Alpha3_scaled = encoded_flat[:, 4]
    
    # Unscale all components
    alpha2_mean = torch.tensor(output_scaler.mean_[2], device=device)
    alpha2_std = torch.tensor(output_scaler.scale_[2], device=device)
    alpha3_mean = torch.tensor(output_scaler.mean_[4], device=device)  
    alpha3_std = torch.tensor(output_scaler.scale_[4], device=device)
    Alpha2 = Alpha2_scaled * alpha2_std + alpha2_mean
    Alpha3 = Alpha3_scaled * alpha3_std + alpha3_mean
    Alpha2 = torch.clamp(Alpha2, -180, 180)
    Alpha3 = torch.clamp(Alpha3, -180, 180)
    
    f1_mean = torch.tensor(output_scaler.mean_[0], device=device)
    f1_std = torch.tensor(output_scaler.scale_[0], device=device)
    f2_mean = torch.tensor(output_scaler.mean_[1], device=device)
    f2_std = torch.tensor(output_scaler.scale_[1], device=device)
    f3_mean = torch.tensor(output_scaler.mean_[3], device=device)
    f3_std = torch.tensor(output_scaler.scale_[3], device=device)
    F1 = F1_scaled * f1_std + f1_mean
    F2 = F2_scaled * f2_std + f2_mean
    F3 = F3_scaled * f3_std + f3_mean
    
    unscaled_commands = torch.stack([F1, F2, Alpha2, F3, Alpha3], dim=1)
    u_max_tensor = torch.tensor([30000, 60000, 180, 60000, 180], device=device)
    violations = torch.clamp(torch.abs(unscaled_commands) - u_max_tensor, min=0.0)
    L2_loss = torch.sum(violations)
    return L2_loss

def ComputeSequenceL3Loss(encoded_commands, output_scaler, device):
    """Compute L3 loss for sequences - now meaningful with temporal data"""
    batch_size, seq_len, _ = encoded_commands.shape
    encoded_flat = encoded_commands.reshape(-1, 5)
    
    # Unscale commands (similar to L2)
    F1_scaled = encoded_flat[:, 0]
    F2_scaled = encoded_flat[:, 1]
    Alpha2_scaled = encoded_flat[:, 2]
    F3_scaled = encoded_flat[:, 3]
    Alpha3_scaled = encoded_flat[:, 4]
    
    alpha2_mean = torch.tensor(output_scaler.mean_[2], device=device)
    alpha2_std = torch.tensor(output_scaler.scale_[2], device=device)
    alpha3_mean = torch.tensor(output_scaler.mean_[4], device=device)  
    alpha3_std = torch.tensor(output_scaler.scale_[4], device=device)
    Alpha2 = Alpha2_scaled * alpha2_std + alpha2_mean
    Alpha3 = Alpha3_scaled * alpha3_std + alpha3_mean
    
    f1_mean = torch.tensor(output_scaler.mean_[0], device=device)
    f1_std = torch.tensor(output_scaler.scale_[0], device=device)
    f2_mean = torch.tensor(output_scaler.mean_[1], device=device)
    f2_std = torch.tensor(output_scaler.scale_[1], device=device)
    f3_mean = torch.tensor(output_scaler.mean_[3], device=device)
    f3_std = torch.tensor(output_scaler.scale_[3], device=device)
    F1 = F1_scaled * f1_std + f1_mean
    F2 = F2_scaled * f2_std + f2_mean
    F3 = F3_scaled * f3_std + f3_mean
    
    unscaled_commands = torch.stack([F1, F2, Alpha2, F3, Alpha3], dim=1)
    
    # Now compute rate changes within sequences
    delta_u_max = torch.tensor([1000, 1000, 10, 1000, 10], device=device)
    
    total_violations = 0
    # Process each sequence separately for rate constraints
    for b in range(batch_size):
        start_idx = b * seq_len
        end_idx = (b + 1) * seq_len
        seq_commands = unscaled_commands[start_idx:end_idx]
        
        if seq_len > 1:
            rate_changes = torch.abs(seq_commands[:-1] - seq_commands[1:])
            violations = torch.clamp(rate_changes - delta_u_max, min=0.0)
            total_violations += torch.sum(violations)
    
    return total_violations

def ComputeSequenceL4Loss(encoded_commands):
    """Compute L4 loss for sequences"""
    encoded_flat = encoded_commands.reshape(-1, 5)
    F1_scaled = encoded_flat[:, 0]
    F2_scaled = encoded_flat[:, 1]
    F3_scaled = encoded_flat[:, 3]
    
    L4_loss = torch.sum(torch.abs(F1_scaled)**1.5 + 
                       torch.abs(F2_scaled)**1.5 + 
                       torch.abs(F3_scaled)**1.5)
    return L4_loss

def ComputeSequenceL5Loss(encoded_commands, output_scaler, device):
    """Compute L5 loss for sequences"""
    encoded_flat = encoded_commands.reshape(-1, 5)
    Alpha2_scaled = encoded_flat[:, 2]
    Alpha3_scaled = encoded_flat[:, 4]
    
    alpha2_mean = torch.tensor(output_scaler.mean_[2], device=device)
    alpha2_std = torch.tensor(output_scaler.scale_[2], device=device)
    alpha3_mean = torch.tensor(output_scaler.mean_[4], device=device)  
    alpha3_std = torch.tensor(output_scaler.scale_[4], device=device)
    Alpha2 = Alpha2_scaled * alpha2_std + alpha2_mean
    Alpha3 = Alpha3_scaled * alpha3_std + alpha3_mean
    Alpha2 = torch.clamp(Alpha2, -180, 180)
    Alpha3 = torch.clamp(Alpha3, -180, 180)
    
    angles = torch.stack([Alpha2, Alpha3], dim=1)
    sector1_violations = ((angles > -100) & (angles < -80)).float()
    sector2_violations = ((angles > 80) & (angles < 100)).float()
    L5_loss = torch.sum(sector1_violations) + torch.sum(sector2_violations)
    return L5_loss
    
def evaluate_model(model, data_loader, output_scaler, device, mode="Validation"):
    model.eval()
    total_loss = 0
    total_l0, total_l1, total_l2, total_l3, total_l4, total_l5 = 0, 0, 0, 0, 0, 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            encoded, decoded = model(batch_x)
            
            # Compute all losses using sequence versions
            L0 = ComputeSequenceL0Loss(encoded, batch_x, output_scaler, device)
            L1 = ComputeSequenceL1Loss(decoded, batch_x)
            L2 = ComputeSequenceL2Loss(encoded, output_scaler, device)
            L3 = ComputeSequenceL3Loss(encoded, output_scaler, device)
            L4 = ComputeSequenceL4Loss(encoded)
            L5 = ComputeSequenceL5Loss(encoded, output_scaler, device)
            
            # Combined loss
            Loss = k0*L0 + k1*L1 + k2*L2 + k3*L3 + k4*L4 + k5*L5
            
            # Accumulate losses
            total_loss += Loss.item()
            total_l0 += L0.item()
            total_l1 += L1.item()
            total_l2 += L2.item()
            total_l3 += L3.item()
            total_l4 += L4.item()
            total_l5 += L5.item()
            num_batches += 1
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_l0 = total_l0 / num_batches
    avg_l1 = total_l1 / num_batches
    avg_l2 = total_l2 / num_batches
    avg_l3 = total_l3 / num_batches
    avg_l4 = total_l4 / num_batches
    avg_l5 = total_l5 / num_batches
    
    print(f"{mode} Results:")
    print(f"  Total Loss: {avg_loss:.6f}")
    print(f"  L0 (Physics): {avg_l0:.6f}")
    print(f"  L1 (Autoencoder): {avg_l1:.6f}")
    print(f"  L2 (Magnitude): {avg_l2:.6f}")
    print(f"  L3 (Rate): {avg_l3:.6f}")
    print(f"  L4 (Power): {avg_l4:.6f}")
    print(f"  L5 (Sector): {avg_l5:.6f}")
    
    return {
        'total_loss': avg_loss,
        'L0': avg_l0, 'L1': avg_l1, 'L2': avg_l2,
        'L3': avg_l3, 'L4': avg_l4, 'L5': avg_l5
    }
        
if __name__ == "__main__":
    # 1. Generate dataset as sequences
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate data
    F1, F2, F3, Alpha2, Alpha3, Tau = GenerateLargeDataset(num_sequences=2000, sequence_length=500)
    
    # Reshape to proper sequences: [num_sequences, seq_length, features]
    num_sequences = 2000
    seq_length = 500
    
    Tau_sequences = Tau.reshape(num_sequences, seq_length, 3)
    commands_flat = torch.stack([F1, F2, Alpha2, F3, Alpha3], dim=1)
    commands_sequences = commands_flat.reshape(num_sequences, seq_length, 5)
    
    # Scale by flattening, scaling, then reshaping back
    tau_flat = Tau_sequences.reshape(-1, 3)
    commands_flat = commands_sequences.reshape(-1, 5)
    
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    
    tau_scaled = input_scaler.fit_transform(tau_flat)
    commands_scaled = output_scaler.fit_transform(commands_flat)
    
    inputs_tensor = torch.tensor(tau_scaled.reshape(num_sequences, seq_length, 3), dtype=torch.float32)
    outputs_tensor = torch.tensor(commands_scaled.reshape(num_sequences, seq_length, 5), dtype=torch.float32)
    
    # Save scalers
    scaler_data = {
        'input_scaler': input_scaler,
        'output_scaler': output_scaler
    }
    with open('scalers.pkl', 'wb') as f:
        pickle.dump(scaler_data, f)
    print("Scalers saved to scalers.pkl")
    
    # 3. Split at sequence level
    total_sequences = len(inputs_tensor)  # 1000
    train_end = int(0.7 * total_sequences)  # 700
    val_end = int(0.8 * total_sequences)    # 800
    
    X_train = inputs_tensor[:train_end]     # [700, 500, 3]
    X_val = inputs_tensor[train_end:val_end] # [100, 500, 3]  
    X_test = inputs_tensor[val_end:]         # [200, 500, 3]
    
    y_train = outputs_tensor[:train_end]     # [700, 500, 5]
    y_val = outputs_tensor[train_end:val_end] # [100, 500, 5]
    y_test = outputs_tensor[val_end:]         # [200, 500, 5]
    
    print(f"Training sequences: {X_train.shape[0]}")
    print(f"Validation sequences: {X_val.shape[0]}")
    print(f"Test sequences: {X_test.shape[0]}")
    
    # Create datasets and dataloaders - shuffle sequences, not points
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Smaller batch sizes for sequence training
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Shuffle sequences
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model with dropout
    model = Autoencoder(hidden_size=64, dropout_rate=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)  # L2 regularization
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Create tensorboard writer
    log_dir = "runs/sequence_control_allocation"
    writer = SummaryWriter(log_dir)
    
    # Training parameters
    num_epochs = 15
    step = 1
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Loss Hyperparameters - increase physics weight, make L3 meaningful
    k0 = 5   # Physics reconstruction
    k1 = 1    # Autoencoder reconstruction  
    k2 = 0.1  # Magnitude constraints
    k3 = 1e-4 # Rate constraints (now meaningful!)
    k4 = 1e-7 # Power consumption
    k5 = 0.1  # Sector constraints
    
    print(f"Starting training with sequence-based approach...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            encoded, decoded = model(batch_x)
            
            # Compute all losses using sequence versions
            L0 = ComputeSequenceL0Loss(encoded, batch_x, output_scaler, device)
            L1 = ComputeSequenceL1Loss(decoded, batch_x)
            L2 = ComputeSequenceL2Loss(encoded, output_scaler, device)
            L3 = ComputeSequenceL3Loss(encoded, output_scaler, device)
            L4 = ComputeSequenceL4Loss(encoded)
            L5 = ComputeSequenceL5Loss(encoded, output_scaler, device)
            
            # Combined Loss
            Loss = k0*L0 + k1*L1 + k2*L2 + k3*L3 + k4*L4 + k5*L5
            
            # Backward pass
            Loss.backward()
            optimizer.step()
            
            # Log to tensorboard
            writer.add_scalar('Loss/L0_Command_Reconstruction', L0.item(), step)
            writer.add_scalar('Loss/L1_Autoencoder', L1.item(), step)
            writer.add_scalar('Loss/L2_Limit_Violation', L2.item(), step)
            writer.add_scalar('Loss/L3_Rate_Limit_Violation', L3.item(), step)
            writer.add_scalar('Loss/L4_Power_consumption', L4.item(), step)
            writer.add_scalar('Loss/L5_Sector_Violations', L5.item(), step)
            writer.add_scalar('Loss/Total_Loss', Loss.item(), step)
            
            total_loss += Loss.item()
            step += 1
            
            if batch_idx % 20 == 0:  # Print more frequently due to fewer batches
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}:')
                print(f'  L0: {L0.item():.6f}, L1: {L1.item():.6f}')
                print(f'  L2: {L2.item():.6f}, L3: {L3.item():.6f}')
                print(f'  L4: {L4.item():.6f}, L5: {L5.item():.6f}')
                print(f'  Total: {Loss.item():.6f}')
        
        # Log epoch-level metrics
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
        print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')
        
        # Validation Phase
        val_results = evaluate_model(model, val_loader, output_scaler, device, "Validation")
        
        # Learning rate scheduling
        scheduler.step(val_results['total_loss'])
        
        # Log epoch-level metrics to tensorboard
        avg_train_loss = total_loss / len(train_loader)
        
        writer.add_scalar('Epoch/Train_Loss', avg_train_loss, epoch)
        writer.add_scalar('Epoch/Val_Loss', val_results['total_loss'], epoch)
        writer.add_scalar('Epoch/Val_L0', val_results['L0'], epoch)
        writer.add_scalar('Epoch/Val_L1', val_results['L1'], epoch)
        writer.add_scalar('Epoch/Val_L2', val_results['L2'], epoch)
        writer.add_scalar('Epoch/Val_L3', val_results['L3'], epoch)
        writer.add_scalar('Epoch/Val_L4', val_results['L4'], epoch)
        writer.add_scalar('Epoch/Val_L5', val_results['L5'], epoch)
        
        print(f'Epoch {epoch} completed:')
        print(f'  Training Loss: {avg_train_loss:.6f}')
        print(f'  Validation Loss: {val_results["total_loss"]:.6f}')
        
        # Save best model
        if val_results['total_loss'] < best_val_loss:
            best_val_loss = val_results['total_loss']
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  New best model saved! (Loss: {best_val_loss:.6f})")
        
        print("-" * 60)

    # Close writer
    writer.close()
    
    # Final Test Evaluation
    print(f"\nLoading best model from epoch {best_epoch}")
    model.load_state_dict(torch.load('best_model.pth'))

    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    test_results = evaluate_model(model, test_loader, output_scaler, device, "Test")

    # Log final test results to tensorboard
    writer = SummaryWriter(log_dir)
    writer.add_scalar('Final/Test_Loss', test_results['total_loss'], 0)
    writer.add_scalar('Final/Test_L0', test_results['L0'], 0)
    writer.add_scalar('Final/Test_L1', test_results['L1'], 0)
    writer.add_scalar('Final/Test_L2', test_results['L2'], 0)
    writer.add_scalar('Final/Test_L3', test_results['L3'], 0)
    writer.add_scalar('Final/Test_L4', test_results['L4'], 0)
    writer.add_scalar('Final/Test_L5', test_results['L5'], 0)
    writer.close()

    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Final test loss: {test_results['total_loss']:.6f}")
    
    # Test sequence-based predictions
    print("\n" + "="*60)
    print("TESTING SEQUENCE PREDICTIONS")
    print("="*60)
    
    with torch.no_grad():
        # Create test sequences - each sequence has the same force repeated
        test_seq_length = 50  # Shorter for testing
        test_sequences = []
        
        test_forces = [[10, 0, 0], [0, 10, 0], [0, 0, 5]]
        
        for forces in test_forces:
            # Create a sequence where each timestep has the same force
            seq = torch.tensor([forces] * test_seq_length, dtype=torch.float32)
            test_sequences.append(seq)
        
        test_sequences = torch.stack(test_sequences)  # [3, 50, 3]
        
        # Scale the sequences
        test_flat = test_sequences.reshape(-1, 3)
        test_scaled = input_scaler.transform(test_flat.numpy())
        test_tensor = torch.tensor(test_scaled.reshape(3, test_seq_length, 3)).to(device)
        
        # Get predictions
        encoded, decoded = model(test_tensor)
        
        # Unscale predictions - take first timestep of each sequence
        first_timestep_commands = encoded[:, 0, :].cpu()  # [3, 5]
        first_timestep_tau = decoded[:, 0, :].cpu()       # [3, 3]
        
        commands_unscaled = output_scaler.inverse_transform(first_timestep_commands.numpy())
        tau_reconstructed = input_scaler.inverse_transform(first_timestep_tau.numpy())
        
        for i, (original_forces, pred_commands, recon_tau) in enumerate(zip(test_forces, commands_unscaled, tau_reconstructed)):
            print(f"\nTest {i+1}: Input Forces {original_forces}")
            print(f"  Predicted Commands: F1={pred_commands[0]:.2f}, F2={pred_commands[1]:.2f}, Alpha2={pred_commands[2]:.1f}°")
            print(f"  F3={pred_commands[3]:.2f}, Alpha3={pred_commands[4]:.1f}°")
            print(f"  Reconstructed Forces: [{recon_tau[0]:.2f}, {recon_tau[1]:.2f}, {recon_tau[2]:.2f}]")