import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

def RandomWalk(Var, Step, minVal, maxVal):
    Var += np.random.uniform(-Step, Step)
    Var = np.clip(Var, minVal, maxVal)
    return Var

def GenerateData(n=10):
    """Generate very short sequences for debugging"""
    F1 = np.zeros(n)
    F1[0] = np.random.uniform(-1000, 1000)  # Smaller range for debugging
    for i in range(1, n):
        F1[i] = RandomWalk(F1[i-1], 50, -1000, 1000)
    
    F2 = np.zeros(n)
    F3 = np.zeros(n)
    F2[0] = np.random.uniform(-500, 500)
    F3[0] = np.random.uniform(-500, 500)
    
    for i in range(1, n):
        F2[i] = RandomWalk(F2[i-1], 25, -500, 500)
        F3[i] = RandomWalk(F3[i-1], 25, -500, 500)
    
    Alpha2 = np.zeros(n)
    Alpha3 = np.zeros(n)
    Alpha2[0] = np.random.uniform(-90, 90)  # Smaller angle range
    Alpha3[0] = np.random.uniform(-90, 90)
    
    for i in range(1, n):
        Alpha2[i] = RandomWalk(Alpha2[i-1], 3, -90, 90)
        Alpha3[i] = RandomWalk(Alpha3[i-1], 3, -90, 90)
    
    return torch.tensor(F1, dtype=torch.float32), torch.tensor(F2, dtype=torch.float32), \
           torch.tensor(F3, dtype=torch.float32), torch.tensor(Alpha2, dtype=torch.float32), \
           torch.tensor(Alpha3, dtype=torch.float32)

def GenerateBmatrix(Alpha2, Alpha3, n=10):
    B = torch.zeros((n, 3, 3))
    l1, l2, l3, l4 = 14.5, 14, 2.7, 2.7
    
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
    F = torch.stack((F1, F2, F3), dim=1)
    Tau = torch.bmm(B, F.unsqueeze(-1)).squeeze(-1)
    return Tau

class SimpleEncoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, output_size=5):  # Much smaller for debugging
        super(SimpleEncoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        print(f"Encoder input shape: {x.shape}")
        out1, (h, c) = self.lstm1(x)
        print(f"LSTM output shape: {out1.shape}")
        print(f"Hidden state shape: {h.shape}, Cell state shape: {c.shape}")
        
        # Apply linear to all timesteps
        batch_size, seq_len, hidden_size = out1.shape
        out1_flat = out1.reshape(-1, hidden_size)
        output_flat = self.linear(out1_flat)
        output = output_flat.reshape(batch_size, seq_len, -1)
        
        print(f"Encoder output shape: {output.shape}")
        print(f"Sample encoder outputs at timestep 0:")
        print(f"  Batch 0: {output[0, 0, :].detach().cpu().numpy()}")
        if batch_size > 1:
            print(f"  Batch 1: {output[1, 0, :].detach().cpu().numpy()}")
        
        return output

class SimpleDecoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=16, output_size=3):
        super(SimpleDecoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        print(f"Decoder input shape: {x.shape}")
        out1, _ = self.lstm1(x)
        
        batch_size, seq_len, hidden_size = out1.shape
        out1_flat = out1.reshape(-1, hidden_size)
        output_flat = self.linear(out1_flat)
        output = output_flat.reshape(batch_size, seq_len, -1)
        
        print(f"Decoder output shape: {output.shape}")
        return output

class DebugAutoencoder(nn.Module):
    def __init__(self):
        super(DebugAutoencoder, self).__init__()
        self.encoder = SimpleEncoder()
        self.decoder = SimpleDecoder()
        
    def forward(self, x):
        print(f"\n--- FORWARD PASS ---")
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        print(f"--- END FORWARD PASS ---\n")
        return encoded, decoded

def debug_loss_computation(encoded, original_tau, output_scaler, device):
    """Debug version of L0 loss with detailed prints"""
    print(f"\n--- DEBUGGING LOSS COMPUTATION ---")
    
    batch_size, seq_len, _ = encoded.shape
    print(f"Encoded shape: {encoded.shape}")
    
    # Flatten
    encoded_flat = encoded.reshape(-1, 5)
    tau_flat = original_tau.reshape(-1, 3)
    print(f"Flattened - Encoded: {encoded_flat.shape}, Tau: {tau_flat.shape}")
    
    # Extract first few samples for debugging
    print(f"First 3 encoded samples (scaled):")
    for i in range(min(3, encoded_flat.shape[0])):
        print(f"  Sample {i}: {encoded_flat[i, :].detach().cpu().numpy()}")
    
    # Unscale (simplified - assume no scaler for now)
    F1 = encoded_flat[:, 0] * 1000  # Simple unscaling for debug
    F2 = encoded_flat[:, 1] * 500
    Alpha2 = encoded_flat[:, 2] * 90
    F3 = encoded_flat[:, 3] * 500
    Alpha3 = encoded_flat[:, 4] * 90
    
    print(f"First 3 unscaled command samples:")
    for i in range(min(3, F1.shape[0])):
        print(f"  Sample {i}: F1={F1[i].item():.2f}, F2={F2[i].item():.2f}, Alpha2={Alpha2[i].item():.1f}°, F3={F3[i].item():.2f}, Alpha3={Alpha3[i].item():.1f}°")
    
    # Build B matrices and compute reconstructed tau
    n = F1.shape[0]
    B = GenerateBmatrix(Alpha2, Alpha3, n).to(device)
    reconstructed_tau = ComputeTau(F1, F2, F3, B)
    
    print(f"First 3 original vs reconstructed tau:")
    for i in range(min(3, tau_flat.shape[0])):
        orig = tau_flat[i, :].detach().cpu().numpy()
        recon = reconstructed_tau[i, :].detach().cpu().numpy()
        print(f"  Sample {i}: Original={orig}, Reconstructed={recon}")
        print(f"             Error={np.abs(orig - recon)}")
    
    # Compute loss
    loss = nn.MSELoss()(reconstructed_tau, tau_flat)
    print(f"L0 Loss: {loss.item()}")
    print(f"--- END LOSS DEBUG ---\n")
    
    return loss

def create_debug_data():
    """Create minimal dataset for debugging"""
    print("=== CREATING DEBUG DATASET ===")
    
    # Generate only 5 sequences of length 10
    num_sequences = 5
    seq_length = 10
    
    all_F1, all_F2, all_F3, all_Alpha2, all_Alpha3, all_Tau = [], [], [], [], [], []
    
    for seq in range(num_sequences):
        print(f"Generating sequence {seq}")
        F1, F2, F3, Alpha2, Alpha3 = GenerateData(seq_length)
        B = GenerateBmatrix(Alpha2, Alpha3, seq_length)
        Tau = ComputeTau(F1, F2, F3, B)
        
        print(f"  Sequence {seq} - Tau range: [{Tau.min().item():.2f}, {Tau.max().item():.2f}]")
        print(f"  Commands - F1: [{F1.min().item():.1f}, {F1.max().item():.1f}]")
        print(f"           - Alpha2: [{Alpha2.min().item():.1f}, {Alpha2.max().item():.1f}]")
        
        all_F1.append(F1)
        all_F2.append(F2)
        all_F3.append(F3)
        all_Alpha2.append(Alpha2)
        all_Alpha3.append(Alpha3)
        all_Tau.append(Tau)
    
    # Concatenate
    F1_final = torch.cat(all_F1)
    F2_final = torch.cat(all_F2)
    F3_final = torch.cat(all_F3)
    Alpha2_final = torch.cat(all_Alpha2)
    Alpha3_final = torch.cat(all_Alpha3)
    Tau_final = torch.cat(all_Tau)
    
    # Reshape to sequences
    Tau_sequences = Tau_final.reshape(num_sequences, seq_length, 3)
    commands_flat = torch.stack([F1_final, F2_final, Alpha2_final, F3_final, Alpha3_final], dim=1)
    commands_sequences = commands_flat.reshape(num_sequences, seq_length, 5)
    
    print(f"Final shapes: Tau={Tau_sequences.shape}, Commands={commands_sequences.shape}")
    
    return Tau_sequences, commands_sequences

def debug_training():
    """Debug training with minimal data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create minimal dataset
    tau_sequences, commands_sequences = create_debug_data()
    
    # Simple scaling (just normalize to [-1, 1] range)
    tau_flat = tau_sequences.reshape(-1, 3)
    commands_flat = commands_sequences.reshape(-1, 5)
    
    # Create simple scalers
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    
    tau_scaled = input_scaler.fit_transform(tau_flat.numpy())
    commands_scaled = output_scaler.fit_transform(commands_flat.numpy())
    
    # Reshape back
    tau_tensor = torch.tensor(tau_scaled.reshape(5, 10, 3), dtype=torch.float32).to(device)
    commands_tensor = torch.tensor(commands_scaled.reshape(5, 10, 5), dtype=torch.float32).to(device)
    
    print(f"Scaled data shapes: Tau={tau_tensor.shape}, Commands={commands_tensor.shape}")
    print(f"Tau range: [{tau_tensor.min().item():.3f}, {tau_tensor.max().item():.3f}]")
    
    # Create model
    model = DebugAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"\n=== STARTING DEBUG TRAINING ===")
    
    # Train for just a few epochs
    for epoch in range(5):
        print(f"\n--- EPOCH {epoch} ---")
        
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with all 5 sequences at once
        encoded, decoded = model(tau_tensor)
        
        # Compute losses
        L0 = debug_loss_computation(encoded, tau_tensor, output_scaler, device)
        L1 = nn.MSELoss()(decoded, tau_tensor)
        
        total_loss = L0 + L1
        
        print(f"Epoch {epoch}: L0={L0.item():.6f}, L1={L1.item():.6f}, Total={total_loss.item():.6f}")
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Print gradient info
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name}: grad_norm={grad_norm:.6f}")
    
    print(f"\n=== TESTING FINAL PREDICTIONS ===")
    model.eval()
    with torch.no_grad():
        # Test with simple inputs
        test_inputs = torch.tensor([
            [[1.0, 0.0, 0.0]] * 10,  # Pure surge sequence
            [[0.0, 1.0, 0.0]] * 10,  # Pure sway sequence  
            [[0.0, 0.0, 1.0]] * 10   # Pure yaw sequence
        ], dtype=torch.float32).to(device)
        
        print(f"Test input shape: {test_inputs.shape}")
        
        encoded, decoded = model(test_inputs)
        
        # Print predictions for first timestep of each sequence
        print(f"\nPredictions at timestep 0:")
        for i in range(3):
            pred = encoded[i, 0, :].cpu().numpy()
            recon = decoded[i, 0, :].cpu().numpy()
            print(f"  Test {i}: Commands={pred}, Reconstructed={recon}")

if __name__ == "__main__":
    debug_training()