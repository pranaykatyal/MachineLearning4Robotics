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
        F1[i] = RandomWalk(F1[i-1],100,-10000,10000)
    
    # Rear Azimuth Thrusters
    F2 = np.zeros(n)
    F3 = np.zeros(n)
    F2[0] = np.random.uniform(-5000, 5000)
    F3[0] = np.random.uniform(-5000, 5000)
    
    for i in range(1, n):
        F2[i] = RandomWalk(F2[i-1], 50, -5000, 5000)
        F3[i] = RandomWalk(F3[i-1], 50, -5000, 5000)
    
    # Azimuth Angles
    Alpha2 = np.zeros(n)
    Alpha3 = np.zeros(n)
    Alpha2[0] = np.random.uniform(-180, 180)
    Alpha3[0] = np.random.uniform(-180, 180)
    
    for i in range(1, n):
        Alpha2[i] = RandomWalk(Alpha2[i-1], 5, -180, 180)
        Alpha3[i] = RandomWalk(Alpha3[i-1], 5, -180, 180)
    
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

def GenerateLargeDataset(num_sequences=1000, sequence_length=1000):
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
    def __init__(self, input_size=3, hidden_size=64, output_size=5):
        super(Encoder, self).__init__()
        # LSTM1, LSTM2, Dense Linear
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        # For now, we'll treat each sample as sequence length 1
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension: [batch, 1, features]
            
        # Pass through LSTM layers
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        
        # Take the last output and pass through linear layer
        output = self.linear(out2[:, -1, :])  # [batch, output_size]
        
        return output    
    
class Decoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=3):
        super(Decoder, self).__init__()
        # LSTM1, LSTM2, Dense Linear
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        # For now, we'll treat each sample as sequence length 1
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension: [batch, 1, features]
            
        # Pass through LSTM layers
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        
        # Take the last output and pass through linear layer
        output = self.linear(out2[:, -1, :])  # [batch, output_size]
        
        return output 

class Autoencoder(nn.Module):
    def __init__(self, hidden_size=64):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size=3, hidden_size=hidden_size, output_size=5)
        self.decoder = Decoder(input_size=5, hidden_size=hidden_size, output_size=3)
        
    def forward(self, x):
        # Encoder: tau (3D) -> thruster commands (5D)
        encoded = self.encoder(x)
        
        # Decoder: thruster commands (5D) -> reconstructed tau (3D)
        decoded = self.decoder(encoded)
        
        return encoded, decoded

def ComputeL0Loss(encoded_commands, original_tau, output_scaler, device):
    # Step 1: Extract components from encoded_commands
    F1_scaled = encoded_commands[:, 0]
    F2_scaled = encoded_commands[:, 1]
    Alpha2_scaled = encoded_commands[:, 2]
    F3_scaled = encoded_commands[:, 3]
    Alpha3_scaled = encoded_commands[:, 4]
    n = F1_scaled.shape[0]
    # Rescale Alpha2 and Alpha3 from scaled values back to original range
    alpha2_mean = torch.tensor(output_scaler.mean_[2], device=device)
    alpha2_std = torch.tensor(output_scaler.scale_[2], device=device)
    alpha3_mean = torch.tensor(output_scaler.mean_[4], device=device)  
    alpha3_std = torch.tensor(output_scaler.scale_[4], device=device)
    # Rescale angles back to original range
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
    # Rescale forces back to original range
    F1 = F1_scaled * f1_std + f1_mean
    F2 = F2_scaled * f2_std + f2_mean
    F3 = F3_scaled * f3_std + f3_mean
    
    # Step 2: Build B matrices
    B = GenerateBmatrix(Alpha2, Alpha3, n).to(device)  # Shape: [n, 3, 3]
      
    # Step 3: Compute reconstructed tau
    reconstructed_tau = ComputeTau(F1, F2, F3, B)  # Shape: [n, 3]
    
    # Step 4: Return MSE loss
    criterion = nn.MSELoss()
    loss_value = criterion(reconstructed_tau, original_tau)
    # print(f"encoded_commands shape: {encoded_commands.shape}")
    # print(f"Alpha2 range: [{Alpha2.min():.2f}, {Alpha2.max():.2f}]")
    # print(f"reconstructed_tau shape: {reconstructed_tau.shape}")
    # print(f"original_tau shape: {original_tau.shape}")
    # In your ComputeL0Loss function, after unscaling:

    
    return loss_value


if __name__ == "__main__":
    # 1. Generate large dataset 
    torch.manual_seed(42)
    np.random.seed(42)
    
    F1, F2, F3, Alpha2, Alpha3, Tau = GenerateLargeDataset(num_sequences=1000, sequence_length=1000)
    thruster_commands = torch.stack([F1, F2, Alpha2, F3, Alpha3], dim=1)  # Shape: [1000000, 5]
    # print(f"Thruster commands shape: {thruster_commands.shape}")
    # print(f"Final shapes:")
    # print(f"Commands: F1={F1.shape}, F2={F2.shape}, F3={F3.shape}, Alpha2={Alpha2.shape}, Alpha3={Alpha3.shape}")
    # print(f"Targets: Tau={Tau.shape}")
    # For control allocation, you want: Tau (input) -> Commands (output)
    inputs = Tau.numpy()           # [1000000, 3] - generalized forces  
    outputs = thruster_commands.numpy()  # [1000000, 5] - thruster commands
    # 2. Scale the Data (standardization)
    # Create scalers
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    
    # Fit and transform
    inputs_scaled = input_scaler.fit_transform(inputs)
    outputs_scaled = output_scaler.fit_transform(outputs)
    
    inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs_scaled, dtype=torch.float32)
    # print(f"Original input range: [{inputs.min():.2f}, {inputs.max():.2f}]")
    # print(f"Scaled input range: [{inputs_scaled.min():.2f}, {inputs_scaled.max():.2f}]")
    # print(f"Scaled input mean: {inputs_scaled.mean(axis=0)}")
    # print(f"Scaled input std: {inputs_scaled.std(axis=0)}")
    
    # 3. Split the Data for Training and Testing
    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        inputs_tensor, 
        outputs_tensor, 
        test_size=0.2, 
        random_state=42
    )

    # print(f"Training set: {X_train.shape[0]:,} samples")
    # print(f"Test set: {X_test.shape[0]:,} samples")
    # print(f"X_train shape: {X_train.shape}")
    # print(f"y_train shape: {y_train.shape}")
    
    #4. Creating Model
    # Create the model -- test model,   Works 
    # model = Autoencoder(hidden_size=64)
    # # Test with a small batch
    # test_input = X_train[:10]  # Take first 10 samples
    # encoded, decoded = model(test_input)
    # print(f"Input shape: {test_input.shape}")
    # print(f"Encoded shape: {encoded.shape}")  # Should be [10, 5]
    # print(f"Decoded shape: {decoded.shape}")  # Should be [10, 3]
    
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    # Set the device to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model, optimizer, and loss
    model = Autoencoder(hidden_size=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Create tensorboard writer
    log_dir = "runs/control_allocation_experiment"
    writer = SummaryWriter(log_dir)

    
    # Basic training loop

    num_epochs = 10
    step = 1
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # Move data to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            encoded, decoded = model(batch_x)
            # loss = criterion(encoded, batch_y)
            # Compute L0 loss
            L0 = ComputeL0Loss(encoded, batch_x, output_scaler, device)
            
            # Backward pass
            L0.backward()
            optimizer.step()
            
            # Log to tensorboard
            writer.add_scalar('Loss/Train_Batch', L0.item(), step)
            
            total_loss += L0.item()
            step += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {L0.item():.4f}')
        
        # Log epoch-level metrics
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
        print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')

    # Close writer
    writer.close()
    