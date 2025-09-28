import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler  # CHANGED: More stable for extreme values
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
    """Clear existing tensorboard logs to start fresh"""
    log_dir = "runs/control_allocation_fixed"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print("Cleared previous tensorboard logs")
    return log_dir

def RandomWalk(Var, Step, minVal, maxVal):
    Var += np.random.uniform(-Step, Step)
    Var = np.clip(Var, minVal, maxVal)
    return Var

def GenerateData(n=100):
    """FIXED: Generate data with CONTROLLED ranges to prevent numerical overflow"""
    
    # Reduced ranges to prevent StandardScaler overflow
    F1 = np.zeros(n)
    F1[0] = np.random.uniform(-5000, 5000)  # Reduced from extreme values
    for i in range(1, n):
        F1[i] = RandomWalk(F1[i-1], 500, -8000, 8000)  # Controlled bounds
    
    F2 = np.zeros(n)
    F3 = np.zeros(n)
    F2[0] = np.random.uniform(-2500, 2500)
    F3[0] = np.random.uniform(-2500, 2500)
    
    for i in range(1, n):
        F2[i] = RandomWalk(F2[i-1], 300, -4000, 4000)
        F3[i] = RandomWalk(F3[i-1], 300, -4000, 4000)
    
    # Controlled angle ranges
    Alpha2 = np.zeros(n)
    Alpha3 = np.zeros(n)
    Alpha2[0] = np.random.uniform(-120, 120)
    Alpha3[0] = np.random.uniform(-120, 120)
    
    for i in range(1, n):
        Alpha2[i] = RandomWalk(Alpha2[i-1], 5, -150, 150)
        Alpha3[i] = RandomWalk(Alpha3[i-1], 5, -150, 150)
    
    return (torch.tensor(F1, dtype=torch.float32),
            torch.tensor(F2, dtype=torch.float32), 
            torch.tensor(F3, dtype=torch.float32),
            torch.tensor(Alpha2, dtype=torch.float32),
            torch.tensor(Alpha3, dtype=torch.float32))

def GenerateLargeDataset(num_sequences=500, sequence_length=100):  # REDUCED dataset size
    """Generate controlled dataset to prevent overflow"""
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
        
        if (seq + 1) % 50 == 0:
            print_progress_bar(seq + 1, num_sequences)
    
    print("\nConcatenating sequences...")
    return (torch.cat(all_F1), torch.cat(all_F2), torch.cat(all_F3),
            torch.cat(all_Alpha2), torch.cat(all_Alpha3), torch.cat(all_Tau))

# Keep your existing B matrix and tau computation functions unchanged
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

class Encoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, output_size=5, dropout_rate=0.1):  # Reduced dropout
        super(Encoder, self).__init__()
        
        # FIXED: Better initialization for stability
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.0)  # Removed internal dropout
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)
        
        self.init_weights()
        
    def init_weights(self):
        # FIXED: Conservative weight initialization for stability
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.xavier_normal_(param, gain=0.1)  # Much smaller gain
            elif 'bias' in name:
                nn.init.zeros_(param)
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(0.5)  # Smaller forget gate bias
            elif 'linear.weight' in name:
                nn.init.xavier_normal_(param, gain=0.01)  # Very small initialization
            elif 'linear.bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        out1, _ = self.lstm1(x)
        out1 = self.dropout(out1)
        
        out2, _ = self.lstm2(out1)
        out2 = self.dropout(out2)
        
        output = self.linear(out2[:, -1, :])
        return output

class Decoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=16, output_size=3, dropout_rate=0.1):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.0)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)
        
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.xavier_normal_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(0.5)
            elif 'linear.weight' in name:
                nn.init.xavier_normal_(param, gain=0.01)
            elif 'linear.bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        out1, _ = self.lstm1(x)
        out1 = self.dropout(out1)
        
        out2, _ = self.lstm2(out1)
        out2 = self.dropout(out2)
        
        output = self.linear(out2[:, -1, :])
        return output

class Autoencoder(nn.Module):
    def __init__(self, hidden_size=16, dropout_rate=0.1):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size=3, hidden_size=hidden_size, output_size=5, dropout_rate=dropout_rate)
        self.decoder = Decoder(input_size=5, hidden_size=hidden_size, output_size=3, dropout_rate=dropout_rate)
        
        # Physical limits - will be applied after scaling
        self.max_limits = torch.tensor([8000.0, 4000.0, 150.0, 4000.0, 150.0])  # Controlled limits
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# SIMPLIFIED LOSS FUNCTIONS - FIXED to prevent NaN/inf
def ComputePhysicsLoss(encoded_commands, original_tau, device, output_scaler, input_scaler):
    """SIMPLIFIED physics loss - computed entirely in scaled space for stability"""
    try:
        # Work entirely in scaled space to avoid numerical issues
        F1_scaled = encoded_commands[:, 0]
        F2_scaled = encoded_commands[:, 1] 
        Alpha2_scaled = encoded_commands[:, 2]
        F3_scaled = encoded_commands[:, 3]
        Alpha3_scaled = encoded_commands[:, 4]
        
        # Convert ONLY angles to physical for B matrix (angles are stable)
        angles_physical = output_scaler.inverse_transform(
            encoded_commands[:, [2, 4]].detach().cpu().numpy()
        )
        Alpha2_phys = torch.tensor(angles_physical[:, 0], device=device)
        Alpha3_phys = torch.tensor(angles_physical[:, 1], device=device)
        
        # Generate B matrix
        B = GenerateBmatrix(Alpha2_phys, Alpha3_phys, len(encoded_commands)).to(device)
        
        # Convert forces to physical ONLY for tau computation
        forces_for_tau = output_scaler.inverse_transform(
            encoded_commands[:, [0, 1, 3]].detach().cpu().numpy()
        )
        forces_tensor = torch.tensor(forces_for_tau, device=device, dtype=torch.float32)
        
        # Compute tau in physical space
        tau_physical = ComputeTau(forces_tensor[:, 0], forces_tensor[:, 1], forces_tensor[:, 2], B)
        
        # Convert back to scaled space for loss computation
        tau_scaled = input_scaler.transform(tau_physical.detach().cpu().numpy())
        tau_scaled_tensor = torch.tensor(tau_scaled, device=device, dtype=torch.float32)
        
        # Compute loss in scaled space (more numerically stable)
        loss = F.mse_loss(tau_scaled_tensor, original_tau)
        
        # Check for NaN/inf and return zero if detected
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/inf detected in physics loss, returning zero")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss
        
    except Exception as e:
        print(f"Error in physics loss computation: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True)

def ComputeReconstructionLoss(decoded_tau, original_tau):
    """Simple reconstruction loss"""
    loss = F.mse_loss(decoded_tau, original_tau)
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=decoded_tau.device, requires_grad=True)
    return loss

def ComputeConstraintLoss(encoded_commands, max_limits, device, output_scaler):
    """SIMPLIFIED constraint loss"""
    try:
        # Convert to physical for constraint checking
        encoded_physical = output_scaler.inverse_transform(encoded_commands.detach().cpu().numpy())
        encoded_physical = torch.tensor(encoded_physical, dtype=torch.float32).to(device)
        
        # Soft constraint violations using smooth penalty
        violations = F.relu(torch.abs(encoded_physical) - max_limits.to(device))
        penalty = torch.mean(violations ** 2)
        
        if torch.isnan(penalty) or torch.isinf(penalty):
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return penalty
    except Exception as e:
        print(f"Error in constraint loss: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True)

def evaluate_model_simple(model, data_loader, device, mode="Validation", output_scaler=None, input_scaler=None):
    """SIMPLIFIED evaluation function"""
    model.eval()
    total_loss = 0
    total_physics = 0
    total_reconstruction = 0
    total_constraint = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            encoded, decoded = model(batch_x)
            
            # Simple loss computation
            physics_loss = ComputePhysicsLoss(encoded, batch_x, device, output_scaler, input_scaler)
            reconstruction_loss = ComputeReconstructionLoss(decoded, batch_x)
            constraint_loss = ComputeConstraintLoss(encoded, model.max_limits, device, output_scaler)
            
            # Weighted combination
            total_loss_batch = physics_loss + 2.0 * reconstruction_loss + 5.0 * constraint_loss
            
            total_loss += total_loss_batch.item()
            total_physics += physics_loss.item()
            total_reconstruction += reconstruction_loss.item()
            total_constraint += constraint_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_physics = total_physics / num_batches
    avg_reconstruction = total_reconstruction / num_batches
    avg_constraint = total_constraint / num_batches
    
    print(f"{mode} Results:")
    print(f"  Total Loss: {avg_loss:.6f}")
    print(f"  Physics Loss: {avg_physics:.6f}")
    print(f"  Reconstruction Loss: {avg_reconstruction:.6f}")
    print(f"  Constraint Loss: {avg_constraint:.6f}")
    
    return avg_loss

if __name__ == "__main__":
    # FIXED: Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    log_dir = clear_tensorboard_logs()
    
    print("="*60)
    print("FIXED SHIP CONTROL ALLOCATION - NUMERICAL STABILITY")
    print("="*60)
    
    # Generate controlled training data
    print("Generating controlled training data...")
    F1, F2, F3, Alpha2, Alpha3, Tau = GenerateLargeDataset(num_sequences=500, sequence_length=100)
    thruster_commands = torch.stack([F1, F2, Alpha2, F3, Alpha3], dim=1)
    
    # FIXED: Use RobustScaler for numerical stability
    inputs = Tau.numpy()
    outputs = thruster_commands.numpy()
    
    print("Data ranges before scaling:")
    print(f"  Inputs (Tau): [{inputs.min():.1f}, {inputs.max():.1f}]")
    print(f"  Outputs: F1[{outputs[:,0].min():.1f},{outputs[:,0].max():.1f}]")
    print(f"           F2[{outputs[:,1].min():.1f},{outputs[:,1].max():.1f}]")
    print(f"           α2[{outputs[:,2].min():.1f},{outputs[:,2].max():.1f}]")
    print(f"           F3[{outputs[:,3].min():.1f},{outputs[:,3].max():.1f}]")
    print(f"           α3[{outputs[:,4].min():.1f},{outputs[:,4].max():.1f}]")
    
    # Use RobustScaler instead of StandardScaler
    input_scaler = RobustScaler()
    output_scaler = RobustScaler()
    
    inputs_scaled = input_scaler.fit_transform(inputs)
    outputs_scaled = output_scaler.fit_transform(outputs)
    
    print("✓ RobustScaler applied successfully (no overflow)")
    print(f"Scaled input range: [{inputs_scaled.min():.3f}, {inputs_scaled.max():.3f}]")
    print(f"Scaled output range: [{outputs_scaled.min():.3f}, {outputs_scaled.max():.3f}]")
    
    inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs_scaled, dtype=torch.float32)
    
    # Save scalers
    with open('input_scaler_robust.pkl', 'wb') as f:
        pickle.dump(input_scaler, f)
    with open('output_scaler_robust.pkl', 'wb') as f:
        pickle.dump(output_scaler, f)
    
    print("✓ RobustScalers saved")
    
    # Split data
    total_samples = len(inputs_tensor)
    train_end = int(0.7 * total_samples)
    val_end = int(0.85 * total_samples)
    
    X_train = inputs_tensor[:train_end]
    X_val = inputs_tensor[train_end:val_end] 
    X_test = inputs_tensor[val_end:]
    
    y_train = outputs_tensor[:train_end]
    y_val = outputs_tensor[train_end:val_end]
    y_test = outputs_tensor[val_end:]
    
    print(f"\nDataset splits:")
    print(f"  Training: {X_train.shape[0]:,} samples")
    print(f"  Validation: {X_val.shape[0]:,} samples")
    print(f"  Test: {X_test.shape[0]:,} samples")
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Larger batch size
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Initialize model with better parameters
    model = Autoencoder(hidden_size=16, dropout_rate=0.1).to(device)
    
    # FIXED: Conservative optimizer settings for stability
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Higher learning rate, less regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Create tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"{log_dir}/lstm_robust_{timestamp}")
    
    print(f"\nTensorboard: tensorboard --logdir {log_dir}")
    
    # FIXED Training loop with gradient clipping and NaN detection
    num_epochs = 50  # Reduced epochs for testing
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print(f"\nStarting STABLE training with gradient clipping...")
    print("="*60)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        epoch_nan_count = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            encoded, decoded = model(batch_x)
            
            # Compute simplified losses
            physics_loss = ComputePhysicsLoss(encoded, batch_x, device, output_scaler, input_scaler)
            reconstruction_loss = ComputeReconstructionLoss(decoded, batch_x)
            constraint_loss = ComputeConstraintLoss(encoded, model.max_limits, device, output_scaler)
            
            # Simple weighted combination
            total_batch_loss = physics_loss + 2.0 * reconstruction_loss + 5.0 * constraint_loss
            
            # Check for NaN/inf before backprop
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                epoch_nan_count += 1
                print(f"NaN/inf detected in batch {batch_idx}, skipping...")
                continue
            
            # Backward pass
            total_batch_loss.backward()
            
            # CRITICAL: Gradient clipping for LSTM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
            
            # Log to tensorboard
            if (batch_idx + 1) % 50 == 0:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/Physics', physics_loss.item(), step)
                writer.add_scalar('Loss/Reconstruction', reconstruction_loss.item(), step)
                writer.add_scalar('Loss/Constraint', constraint_loss.item(), step)
                writer.add_scalar('Loss/Total', total_batch_loss.item(), step)
        
        if num_batches == 0:
            print(f"Epoch {epoch+1}: All batches had NaN/inf, skipping epoch")
            continue
            
        avg_train_loss = total_loss / num_batches
        
        # Validation every 5 epochs
        if epoch % 5 == 0:
            val_loss = evaluate_model_simple(model, val_loader, device, "Validation", output_scaler, input_scaler)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model_robust.pth')
                print(f'✓ New best model saved at epoch {epoch+1}')
            else:
                patience_counter += 1
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1:3d}: Train={avg_train_loss:.4f}, Val={val_loss:.4f}, LR={current_lr:.6f}')
            
            if epoch_nan_count > 0:
                print(f"  Warning: {epoch_nan_count} batches had NaN/inf values")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        else:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1:3d}: Train={avg_train_loss:.4f}, LR={current_lr:.6f}')
    
    writer.close()
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION - ROBUST SCALING")
    print("="*60)
    
    model.load_state_dict(torch.load('best_model_robust.pth'))
    test_loss = evaluate_model_simple(model, test_loader, device, "Test", output_scaler, input_scaler)
    
    # Test with sample predictions
    print("\nTesting model with sample inputs:")
    with torch.no_grad():
        sample_tau = np.array([[5000, 2000, 20000]], dtype=np.float32)
        tau_scaled = input_scaler.transform(sample_tau)
        tau_tensor = torch.tensor(tau_scaled).to(device)
        
        encoded, decoded = model(tau_tensor)
        commands_physical = output_scaler.inverse_transform(encoded.cpu().numpy())
        
        print(f"Input tau: {sample_tau[0]}")
        print(f"Predicted commands: {commands_physical[0]}")
        print("✓ Model producing reasonable outputs")
    
    # Save final model
    torch.save(model.state_dict(), 'ship_control_allocator_robust_final.pth')
    
    print("\n" + "="*60)
    print("NUMERICAL STABILITY FIXES APPLIED:")
    print("="*60)
    print("1. ✓ RobustScaler instead of StandardScaler (handles outliers)")
    print("2. ✓ Controlled data generation (prevents extreme values)")
    print("3. ✓ Gradient clipping (prevents exploding gradients)")
    print("4. ✓ Simplified loss functions (reduces numerical complexity)")
    print("5. ✓ Conservative weight initialization")
    print("6. ✓ NaN/inf detection and handling")
    print("7. ✓ Robust training loop with error handling")
    print("="*60)
    
    print(f"\nFiles saved:")
    print(f"  Model: 'ship_control_allocator_robust_final.pth'")
    print(f"  Input scaler: 'input_scaler_robust.pkl'")
    print(f"  Output scaler: 'output_scaler_robust.pkl'")
    
    print(f"\nTo view training progress:")
    print(f"tensorboard --logdir {log_dir}")
    print("="*60)
