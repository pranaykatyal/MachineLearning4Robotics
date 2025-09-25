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
    def __init__(self, input_size=3, hidden_size=16, output_size=5):
        super(SimpleEncoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
    def forward(self, x):
        print(f"Encoder input shape: {x.shape}")
        out1, (h, c) = self.lstm1(x)
        batch_size, seq_len, hidden_size = out1.shape
        out1_flat = out1.reshape(-1, hidden_size)
        output_flat = self.linear(out1_flat)
        output = output_flat.reshape(batch_size, seq_len, -1)
        output = torch.clamp(output, -10.0, 10.0)
        print(f"Encoder output shape: {output.shape}")
        return output

class SimpleDecoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=16, output_size=3):
        super(SimpleDecoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out1, _ = self.lstm1(x)
        batch_size, seq_len, hidden_size = out1.shape
        out1_flat = out1.reshape(-1, hidden_size)
        output_flat = self.linear(out1_flat)
        output = output_flat.reshape(batch_size, seq_len, -1)
        return output

class DebugAutoencoder(nn.Module):
    def __init__(self):
        super(DebugAutoencoder, self).__init__()
        self.encoder = SimpleEncoder()
        self.decoder = SimpleDecoder()
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class FixedPhysicsAwareEncoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, output_size=5):
        super(FixedPhysicsAwareEncoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=1.0)  # Increased gain
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        print(f"FixedPhysicsEncoder input shape: {x.shape}")
        out1, (h, c) = self.lstm1(x)
        batch_size, seq_len, hidden_size = out1.shape
        out1_flat = out1.reshape(-1, hidden_size)
        output_flat = self.linear(out1_flat)
        output = output_flat.reshape(batch_size, seq_len, -1)
        
        # FIXED: Direct scaling without tanh bottleneck and no in-place operations
        # Based on your actual command ranges: [-451, 935] for forces, [-90, 90] for angles
        
        # Scale outputs to match your actual data ranges - create new tensors, don't modify in place
        F1 = output[:, :, 0] * 500    # F1: roughly ±1000 range
        F2 = output[:, :, 1] * 300    # F2: roughly ±600 range  
        Alpha2 = torch.clamp(output[:, :, 2] * 45, -90, 90)    # Alpha2: [-90, 90]
        F3 = output[:, :, 3] * 300    # F3: roughly ±600 range
        Alpha3 = torch.clamp(output[:, :, 4] * 45, -90, 90)    # Alpha3: [-90, 90]
        
        # Stack the results into a new tensor
        output = torch.stack([F1, F2, Alpha2, F3, Alpha3], dim=-1)
        
        print(f"FixedPhysicsEncoder output shape: {output.shape}")
        print(f"Sample physics outputs at timestep 0:")
        print(f"  Batch 0: F1={output[0, 0, 0]:.1f}, F2={output[0, 0, 1]:.1f}, A2={output[0, 0, 2]:.1f}°, F3={output[0, 0, 3]:.1f}, A3={output[0, 0, 4]:.1f}°")
        if batch_size > 1:
            print(f"  Batch 1: F1={output[1, 0, 0]:.1f}, F2={output[1, 0, 1]:.1f}, A2={output[1, 0, 2]:.1f}°, F3={output[1, 0, 3]:.1f}, A3={output[1, 0, 4]:.1f}°")
        
        return output

class AdaptivePhysicsEncoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=5):
        super(AdaptivePhysicsEncoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
        # Learnable output scaling parameters
        self.force_scale = nn.Parameter(torch.tensor([800.0, 400.0, 400.0]))  # F1, F2, F3
        self.angle_scale = nn.Parameter(torch.tensor([60.0, 60.0]))  # Alpha2, Alpha3
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        print(f"AdaptiveEncoder input shape: {x.shape}")
        out1, (h, c) = self.lstm1(x)
        batch_size, seq_len, hidden_size = out1.shape
        out1_flat = out1.reshape(-1, hidden_size)
        output_flat = self.linear(out1_flat)
        output = output_flat.reshape(batch_size, seq_len, -1)
        
        # Adaptive scaling with learnable parameters - no in-place operations
        F1 = output[:, :, 0] * self.force_scale[0]    # F1
        F2 = output[:, :, 1] * self.force_scale[1]    # F2  
        Alpha2 = torch.clamp(output[:, :, 2] * self.angle_scale[0], -90, 90)    # Alpha2
        F3 = output[:, :, 3] * self.force_scale[2]    # F3
        Alpha3 = torch.clamp(output[:, :, 4] * self.angle_scale[1], -90, 90)    # Alpha3
        
        # Stack the results into a new tensor
        output = torch.stack([F1, F2, Alpha2, F3, Alpha3], dim=-1)
        
        print(f"AdaptiveEncoder output shape: {output.shape}")
        print(f"Learned scales: F_scales={self.force_scale.data}, A_scales={self.angle_scale.data}")
        print(f"Sample outputs: F1={output[0, 0, 0]:.1f}, F2={output[0, 0, 1]:.1f}, A2={output[0, 0, 2]:.1f}°")
        
        return output

class FixedPhysicsAwareAutoencoder(nn.Module):
    def __init__(self, use_adaptive=True):
        super(FixedPhysicsAwareAutoencoder, self).__init__()
        if use_adaptive:
            self.encoder = AdaptivePhysicsEncoder()
        else:
            self.encoder = FixedPhysicsAwareEncoder()
        self.decoder = SimpleDecoder()
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def create_physics_consistent_data():
    """Create dataset where tau values correspond to optimal achievable thruster configurations"""
    print("=== CREATING PHYSICS-CONSISTENT DATASET ===")
    
    num_sequences = 5
    seq_length = 10
    
    all_Tau, all_Commands = [], []
    
    # Method 1: Generate tau first, then solve for optimal commands
    for seq in range(num_sequences):
        print(f"Generating sequence {seq}")
        
        tau_sequence = []
        command_sequence = []
        
        for step in range(seq_length):
            # Generate desired generalized forces (what motion controller would request)
            # These should be in realistic ranges for ship operations
            surge_force = np.random.uniform(-2000, 2000)    # Surge force [N]
            sway_force = np.random.uniform(-1000, 1000)     # Sway force [N] 
            yaw_moment = np.random.uniform(-5000, 5000)     # Yaw moment [N⋅m]
            
            desired_tau = torch.tensor([surge_force, sway_force, yaw_moment], dtype=torch.float32)
            
            # Find optimal thruster configuration for this tau using analytical solution
            optimal_commands = solve_optimal_allocation(desired_tau)
            
            tau_sequence.append(desired_tau)
            command_sequence.append(optimal_commands)
        
        all_Tau.append(torch.stack(tau_sequence))
        all_Commands.append(torch.stack(command_sequence))
        
        # Print statistics
        tau_tensor = torch.stack(tau_sequence)
        cmd_tensor = torch.stack(command_sequence)
        print(f"  Sequence {seq} - Tau range: [{tau_tensor.min().item():.2f}, {tau_tensor.max().item():.2f}]")
        print(f"  Commands - F1: [{cmd_tensor[:, 0].min().item():.1f}, {cmd_tensor[:, 0].max().item():.1f}]")
        print(f"           - Alpha2: [{cmd_tensor[:, 2].min().item():.1f}, {cmd_tensor[:, 2].max().item():.1f}]")
    
    # Stack all sequences
    Tau_sequences = torch.stack(all_Tau)        # [num_seq, seq_len, 3]
    Commands_sequences = torch.stack(all_Commands)  # [num_seq, seq_len, 5]
    
    print(f"Final shapes: Tau={Tau_sequences.shape}, Commands={Commands_sequences.shape}")
    
    return Tau_sequences, Commands_sequences

def solve_optimal_allocation(desired_tau):
    """Solve for optimal thruster allocation given desired tau"""
    # This implements a simple analytical solution for your 3-thruster system
    # In practice, you'd use optimization, but this gives a reasonable approximation
    
    tau_x, tau_y, tau_n = desired_tau[0].item(), desired_tau[1].item(), desired_tau[2].item()
    
    # Thruster geometry parameters
    l1, l2, l3, l4 = 14.5, 14, 2.7, 2.7
    
    # Method: Use simplified allocation strategy
    
    # Strategy 1: Try to minimize thruster usage while meeting force requirements
    
    # For yaw-dominant requests, use azimuth thrusters efficiently
    if abs(tau_n) > abs(tau_x) + abs(tau_y):
        # Yaw-dominant: set angles for maximum yaw efficiency
        alpha2 = 45.0 if tau_n > 0 else -45.0
        alpha3 = -45.0 if tau_n > 0 else 45.0
    
    # For surge-dominant requests  
    elif abs(tau_x) > abs(tau_y):
        # Surge-dominant: align thrusters with X-axis
        alpha2 = 0.0
        alpha3 = 0.0
    
    # For sway-dominant requests
    else:
        # Sway-dominant: align thrusters with Y-axis
        alpha2 = 90.0 if tau_y > 0 else -90.0
        alpha3 = 90.0 if tau_y > 0 else -90.0
    
    # Clamp angles to reasonable range
    alpha2 = np.clip(alpha2, -90, 90)
    alpha3 = np.clip(alpha3, -90, 90)
    
    # Compute B matrix for these angles
    alpha2_tensor = torch.tensor([alpha2], dtype=torch.float32)
    alpha3_tensor = torch.tensor([alpha3], dtype=torch.float32)
    B = GenerateBmatrix(alpha2_tensor, alpha3_tensor, 1)[0]  # Get first (and only) matrix
    
    # Solve for forces using pseudo-inverse: F = B† × tau
    try:
        B_pinv = torch.pinverse(B)  # Pseudo-inverse for least-squares solution
        forces = torch.matmul(B_pinv, desired_tau.unsqueeze(-1)).squeeze(-1)
        
        F1, F2, F3 = forces[0].item(), forces[1].item(), forces[2].item()
        
        # Clamp forces to realistic ranges
        F1 = np.clip(F1, -2000, 2000)
        F2 = np.clip(F2, -1000, 1000) 
        F3 = np.clip(F3, -1000, 1000)
        
    except:
        # Fallback: use simple heuristic allocation
        F1 = tau_x * 0.5  # Tunnel thruster contributes to surge
        F2 = tau_y * 0.5  # Azimuth thrusters contribute to sway  
        F3 = tau_y * 0.5
        
        F1 = np.clip(F1, -2000, 2000)
        F2 = np.clip(F2, -1000, 1000)
        F3 = np.clip(F3, -1000, 1000)
    
    # Return command vector: [F1, F2, Alpha2, F3, Alpha3]
    commands = torch.tensor([F1, F2, alpha2, F3, alpha3], dtype=torch.float32)
    
    return commands

def debug_training_fixed():
    """Training with properly scaled physics-aware encoder"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create minimal dataset
    tau_sequences, commands_sequences = create_physics_consistent_data()
    
    print(f"BEFORE SCALING:")
    print(f"Tau range: [{tau_sequences.min().item():.2f}, {tau_sequences.max().item():.2f}]")
    print(f"Commands range: [{commands_sequences.min().item():.2f}, {commands_sequences.max().item():.2f}]")
    
    # Scale only tau inputs
    tau_flat = tau_sequences.reshape(-1, 3)
    input_scaler = StandardScaler()
    tau_scaled = input_scaler.fit_transform(tau_flat.numpy())
    
    tau_tensor = torch.tensor(tau_scaled.reshape(5, 10, 3), dtype=torch.float32).to(device)
    commands_tensor = commands_sequences.to(device)
    
    print(f"AFTER SCALING:")
    print(f"Tau (scaled): [{tau_tensor.min().item():.3f}, {tau_tensor.max().item():.3f}]")
    print(f"Commands (original): [{commands_tensor.min().item():.2f}, {commands_tensor.max().item():.2f}]")
    
    # Create FIXED physics-aware model
    model = FixedPhysicsAwareAutoencoder(use_adaptive=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    print(f"\n=== STARTING FIXED PHYSICS-AWARE TRAINING ===")
    
    def physics_loss(encoded, original_tau):
        batch_size, seq_len, _ = encoded.shape
        encoded_flat = encoded.reshape(-1, 5)
        tau_flat = original_tau.reshape(-1, 3)
        
        # Extract commands (already in correct units)
        F1 = encoded_flat[:, 0]
        F2 = encoded_flat[:, 1]
        Alpha2 = encoded_flat[:, 2]
        F3 = encoded_flat[:, 3]
        Alpha3 = encoded_flat[:, 4]
        
        print(f"Command ranges: F1=[{F1.min():.1f},{F1.max():.1f}], Alpha2=[{Alpha2.min():.1f},{Alpha2.max():.1f}]")
        
        # Physics computation
        n = F1.shape[0]
        B = GenerateBmatrix(Alpha2, Alpha3, n).to(device)
        reconstructed_tau = ComputeTau(F1, F2, F3, B)
        
        # Unscale tau_flat to compare in original units
        tau_original = input_scaler.inverse_transform(tau_flat.detach().cpu().numpy())
        tau_original_tensor = torch.tensor(tau_original, device=device)
        
        loss = nn.MSELoss()(reconstructed_tau, tau_original_tensor)
        
        print(f"Sample comparison (original units):")
        print(f"  Target: {tau_original_tensor[0].cpu().numpy()}")
        print(f"  Predicted: {reconstructed_tau[0].detach().cpu().numpy()}")
        print(f"  Physics loss: {loss.item():.6f}")
        
        return loss * 0.01  # Small scaling factor
    
    # Training loop
    for epoch in range(20):  # More epochs since we have better scaling
        print(f"\n--- EPOCH {epoch} ---")
        
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        encoded, decoded = model(tau_tensor)
        
        # Compute losses
        L0 = physics_loss(encoded, tau_tensor)
        L1 = nn.MSELoss()(decoded, tau_tensor)
        
        total_loss = L0 + L1
        
        print(f"Epoch {epoch}: L0={L0.item():.6f}, L1={L1.item():.6f}, Total={total_loss.item():.6f}")
        
        # Early stopping if physics loss gets reasonable
        if L0.item() < 10.0:
            print("Physics loss reached reasonable level!")
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Print max gradient
        max_grad = max(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"  Max gradient norm: {max_grad:.6f}")
    
    # Physics verification test
    print(f"\n=== FINAL PHYSICS TEST ===")
    model.eval()
    with torch.no_grad():
        simple_tau = torch.tensor([[
            [0.0, 0.0, 1000.0],  # Pure yaw moment
            [1000.0, 0.0, 0.0], # Pure surge force
            [0.0, 1000.0, 0.0], # Pure sway force
        ]], dtype=torch.float32)
        
        simple_tau_scaled = input_scaler.transform(simple_tau.reshape(-1, 3))
        simple_tau_tensor = torch.tensor(simple_tau_scaled.reshape(1, 3, 3), dtype=torch.float32).to(device)
        
        encoded, decoded = model(simple_tau_tensor)
        
        print(f"Physics verification:")
        for i in range(3):
            commands = encoded[0, i, :].cpu().numpy()
            F1, F2, Alpha2, F3, Alpha3 = commands
            print(f"  Input tau: {simple_tau[0, i, :].numpy()}")
            print(f"  Predicted commands: F1={F1:.1f}, F2={F2:.1f}, A2={Alpha2:.1f}°, F3={F3:.1f}, A3={Alpha3:.1f}°")
            
            # Verify by computing tau back
            B_test = GenerateBmatrix(torch.tensor([Alpha2]), torch.tensor([Alpha3]), 1)
            tau_verify = ComputeTau(torch.tensor([F1]), torch.tensor([F2]), torch.tensor([F3]), B_test)
            print(f"  Reconstructed tau: {tau_verify[0].numpy()}")
            error = torch.abs(tau_verify[0] - torch.tensor(simple_tau[0, i, :]))
            print(f"  Error: {error.numpy()}")
            print()

def quick_infinity_check():
    """Quick test to see if we can reproduce the infinity issue"""
    print("\n" + "="*50)
    print("QUICK INFINITY CHECK")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test 1: Check what your model outputs initially
    print("1. Testing untrained model outputs:")
    model = DebugAutoencoder().to(device)
    
    test_input = torch.randn(2, 5, 3).to(device)
    print(f"Test input range: [{test_input.min().item():.3f}, {test_input.max().item():.3f}]")
    
    with torch.no_grad():
        try:
            encoded, decoded = model(test_input)
            print(f"Encoded range: [{encoded.min().item():.3f}, {encoded.max().item():.3f}]")
            print(f"Any inf in encoded? {torch.isinf(encoded).any()}")
            print(f"Any nan in encoded? {torch.isnan(encoded).any()}")
        except Exception as e:
            print(f"Error in model forward pass: {e}")
    
    # Test 2: Test physics computation with extreme values
    print("\n2. Testing physics computation with extreme values:")
    extreme_angles = torch.tensor([-500, -90, 0, 90, 500], dtype=torch.float32)
    extreme_forces = torch.tensor([50000, 10000, 0, -10000, -50000], dtype=torch.float32)
    
    try:
        B = GenerateBmatrix(extreme_angles, extreme_angles, 5)
        tau = ComputeTau(extreme_forces, extreme_forces, extreme_forces, B)
        print(f"Max tau with extreme inputs: {torch.abs(tau).max().item():.2e}")
        
        if torch.abs(tau).max() > 1e6:
            print("WARNING: Physics computation produces huge values!")
        else:
            print("Physics computation looks reasonable")
            
    except Exception as e:
        print(f"ERROR in physics computation: {e}")

if __name__ == "__main__":
    # First run the infinity check
    quick_infinity_check()
    
    # Then run the fixed physics-aware training
    print("\n" + "="*60)
    print("RUNNING FIXED PHYSICS-AWARE TRAINING")
    print("="*60)
    try:
        debug_training_fixed()
    except Exception as e:
        print(f"Error in physics-aware training: {e}")
        import traceback
        traceback.print_exc()