import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from dubinEHF3d import dubinEHF3d
from scipy.io import savemat, loadmat  # <-- Added for .mat support

# ===============================
# Configuration
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

dtype = torch.float32

# Hyperparameters (tuned for local GPU)
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50  # Reduced for faster local testing
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2

# Dataset generation (reduced for faster testing)
TURN_RADIUS = 50
PATH_STEP_SIZE = 10
HEADING_INCREMENT = 10  # Coarser: 0°, 20°, 40°, ..., 340° (18 values)
CLIMB_ANGLE_RANGE = 30
CLIMB_ANGLE_INCREMENT = 5  # Coarser: -30°, -20°, -10°, 0°, 10°, 20°, 30° (7 values)
GRID_MARGIN = 5
GRID_SIZE = 2 * TURN_RADIUS * GRID_MARGIN
GRID_RESOLUTION = 5  # Coarser grid for faster generation

MAX_SEQ_LENGTH = 50

# ===============================
# Dataset Generation
# ===============================
def generate_dataset():
    """Generate Dubins trajectory dataset with metadata."""
    dataset = []
    data_count = 0
    
    print("\n" + "="*60)
    print("GENERATING DATASET")
    print("="*60)
    
    for climb_angle_deg in range(-CLIMB_ANGLE_RANGE, CLIMB_ANGLE_RANGE + 1, CLIMB_ANGLE_INCREMENT):
        climb_angle = climb_angle_deg * (np.pi / 180)
        
        for start_heading_deg in range(0, 360, HEADING_INCREMENT):
            start_heading = start_heading_deg * (np.pi / 180)
            
            for x_goal in range(-GRID_SIZE//2, GRID_SIZE//2 + 1, GRID_RESOLUTION):
                for y_goal in range(-GRID_SIZE//2, GRID_SIZE//2 + 1, GRID_RESOLUTION):
                    # Skip origin
                    if x_goal == 0 and y_goal == 0:
                        continue
                    
                    data_count += 1
                    path, end_heading, num_points = dubinEHF3d(
                        0, 0, 0,  # Start position
                        start_heading,
                        x_goal, y_goal,
                        TURN_RADIUS,
                        PATH_STEP_SIZE,
                        climb_angle,
                        data_count
                    )
                    
                    # Only keep valid paths
                    if num_points > 0:
                        trajectory = path[:num_points]
                        final_pos = trajectory[-1]
                        metadata = np.array([
                            0, 0, 0,
                            final_pos[0], final_pos[1], final_pos[2],
                            start_heading,
                            climb_angle
                        ])
                        dataset.append({
                            'trajectory': trajectory,
                            'metadata': metadata,
                            'num_points': num_points
                        })
    print(f"\n✓ Generated {len(dataset)} valid trajectories")
    print("="*60)
    # Save to Data.mat
    print("Saving dataset to Data.mat ...")
    # Convert to arrays for .mat
    trajectories = [d['trajectory'] for d in dataset]
    metadatas = [d['metadata'] for d in dataset]
    num_points = [d['num_points'] for d in dataset]
    savemat('Data.mat', {
        'trajectories': np.array(trajectories, dtype=object),
        'metadatas': np.array(metadatas),
        'num_points': np.array(num_points)
    })
    print("✓ Saved Data.mat")
    return dataset

def load_dataset_from_mat(mat_path='Data.mat'):
    """Load dataset from Data.mat file."""
    print(f"Loading dataset from {mat_path} ...")
    mat = loadmat(mat_path)
    trajectories = mat['trajectories']
    metadatas = mat['metadatas']
    num_points = mat['num_points'].flatten()
    dataset = []
    for i in range(len(num_points)):
        traj = np.array(trajectories[i])
        meta = np.array(metadatas[i])
        npts = int(num_points[i])
        dataset.append({
            'trajectory': traj,
            'metadata': meta,
            'num_points': npts
        })
    print(f"✓ Loaded {len(dataset)} samples from {mat_path}")
    return dataset

# ===============================
# LSTM Encoder-Decoder Model
# ===============================
class DubinsLSTM(nn.Module):
    """LSTM Encoder-Decoder for Dubins trajectory prediction."""
    
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=3, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Encoder: processes initial conditions
        # Output must be 2 * (hidden_dim * num_layers) for h0 and c0
        encoder_output_dim = 2 * hidden_dim * num_layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, encoder_output_dim),
            nn.ReLU()
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def init_hidden(self, batch_size, encoded):
        """Initialize LSTM hidden states from encoded features."""
        hidden_size = self.hidden_dim * self.num_layers
        h0 = encoded[:, :hidden_size].reshape(batch_size, self.num_layers, self.hidden_dim)
        c0 = encoded[:, hidden_size:].reshape(batch_size, self.num_layers, self.hidden_dim)
        
        h0 = h0.transpose(0, 1).contiguous()
        c0 = c0.transpose(0, 1).contiguous()
        
        return (h0, c0)
    
    def forward(self, metadata, target_seq_len, target_trajectory=None, teacher_forcing_ratio=0.5):
        """
        Forward pass with optional teacher forcing.
        
        Args:
            metadata: (batch, 7) - initial conditions
            target_seq_len: int - length of sequence to generate
            target_trajectory: (batch, seq_len, 3) - ground truth for teacher forcing
            teacher_forcing_ratio: probability of using teacher forcing
        """
        batch_size = metadata.shape[0]
        
        # Encode initial conditions
        encoded = self.encoder(metadata)
        hidden = self.init_hidden(batch_size, encoded)
        
        # Start from origin
        decoder_input = torch.zeros(batch_size, 1, self.output_dim).to(metadata.device)
        decoder_input[:, 0, :] = metadata[:, :3]  # Start position
        
        outputs = []
        
        for t in range(target_seq_len):
            lstm_out, hidden = self.decoder_lstm(decoder_input, hidden)
            prediction = self.fc_out(lstm_out)
            outputs.append(prediction)
            
            # Teacher forcing
            if target_trajectory is not None and np.random.random() < teacher_forcing_ratio:
                decoder_input = target_trajectory[:, t:t+1, :]
            else:
                decoder_input = prediction
        
        return torch.cat(outputs, dim=1)

# ===============================
# Training Function
# ===============================
def train_model(dataset):
    """Train the LSTM model with TensorBoard logging."""
    
    # Setup TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/dubins_lstm_{timestamp}"
    writer = SummaryWriter(log_dir)
    
    print(f"\n📊 TensorBoard logging to: {log_dir}")
    print("   Run: tensorboard --logdir=runs --port=6006")
    
    # Prepare data
    print("\n" + "="*60)
    print("PREPARING DATA")
    print("="*60)
    
    trajectories = []
    metadatas = []
    seq_lengths = []
    
    for sample in dataset:
        traj = torch.tensor(sample['trajectory'], dtype=dtype)
        meta = torch.tensor(sample['metadata'], dtype=dtype)
        
        # Pad or truncate
        if len(traj) > MAX_SEQ_LENGTH:
            traj = traj[:MAX_SEQ_LENGTH]
            seq_len = MAX_SEQ_LENGTH
        else:
            padding = torch.zeros(MAX_SEQ_LENGTH - len(traj), 3)
            traj = torch.cat([traj, padding], dim=0)
            seq_len = sample['num_points']
        
        trajectories.append(traj)
        metadatas.append(meta)
        seq_lengths.append(seq_len)
    
    trajectories = torch.stack(trajectories)
    metadatas = torch.stack(metadatas)
    seq_lengths = torch.tensor(seq_lengths)
    
    # Create dataset
    full_dataset = TensorDataset(metadatas, trajectories, seq_lengths)
    
    # Split: 80/10/10
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print("="*60)
    
    # Initialize model
    model = DubinsLSTM(
        input_dim=8,  # [x0, y0, z0, x_goal, y_goal, z_goal, heading, climb]
        hidden_dim=HIDDEN_DIM,
        output_dim=3,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(model)
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*60)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    
    print("\nOptimizer: Adam")
    print(f"Learning Rate: {LR}")
    print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    
    # Log hyperparameters
    hparams = {
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
    }
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 15
    
    train_losses = []
    val_losses = []
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (metadata, trajectory, seq_len) in enumerate(train_loader):
            metadata = metadata.to(device)
            trajectory = trajectory.to(device)
            
            optimizer.zero_grad()
            
            # Teacher forcing schedule (decreases over time)
            teacher_forcing_ratio = max(0.5 * (1 - epoch / NUM_EPOCHS), 0.1)
            max_len = int(seq_len.max().item())
            
            predictions = model(metadata, max_len, trajectory, teacher_forcing_ratio)
            
            # Calculate loss only on valid timesteps
            loss = 0
            for i in range(len(seq_len)):
                valid_len = int(seq_len[i].item())
                loss += criterion(predictions[i, :valid_len], trajectory[i, :valid_len])
            loss /= len(seq_len)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Log batch metrics
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for metadata, trajectory, seq_len in val_loader:
                metadata = metadata.to(device)
                trajectory = trajectory.to(device)
                
                max_len = int(seq_len.max().item())
                predictions = model(metadata, max_len, None, 0.0)
                
                loss = 0
                for i in range(len(seq_len)):
                    valid_len = int(seq_len[i].item())
                    loss += criterion(predictions[i, :valid_len], trajectory[i, :valid_len])
                loss /= len(seq_len)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Log epoch metrics
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Teacher_Forcing_Ratio', teacher_forcing_ratio, epoch)
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"  → Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # Print progress
        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] "
              f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"TF: {teacher_forcing_ratio:.2f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'best_dubins_lstm.pth')
            print(f"  ★ Best model saved! Val Loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break
    
    print("="*60)
    
    # Test evaluation
    print("\n" + "="*60)
    print("TESTING")
    print("="*60)
    
    checkpoint = torch.load('best_dubins_lstm.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loss = 0.0
    with torch.no_grad():
        for metadata, trajectory, seq_len in test_loader:
            metadata = metadata.to(device)
            trajectory = trajectory.to(device)
            
            max_len = int(seq_len.max().item())
            predictions = model(metadata, max_len, None, 0.0)
            
            loss = 0
            for i in range(len(seq_len)):
                valid_len = int(seq_len[i].item())
                loss += criterion(predictions[i, :valid_len], trajectory[i, :valid_len])
            loss /= len(seq_len)
            
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print("="*60)
    
    writer.add_scalar('Loss/test', avg_test_loss, 0)
    
    # Log final hyperparameters with test metric
    writer.add_hparams(
        hparams,
        {
            'hparam/test_loss': avg_test_loss,
            'hparam/best_val_loss': best_val_loss,
            'hparam/final_train_loss': train_losses[-1]
        }
    )
    
    writer.close()
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.axhline(y=avg_test_loss, color='r', linestyle='--', label=f'Test Loss ({avg_test_loss:.2f})')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Loss curves saved: training_curves.png")
    
    return model, test_dataset

# ===============================
# Visualization
# ===============================
def plot_predictions(model, test_dataset, num_samples=10):
    """Generate prediction plots."""
    save_dir = "plots/test_predictions"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"GENERATING {num_samples} PREDICTION PLOTS")
    print("="*60)
    
    model.eval()
    
    for i in range(min(num_samples, len(test_dataset))):
        metadata, trajectory, seq_len = test_dataset[i]
        
        metadata_input = metadata.unsqueeze(0).to(device)
        valid_len = int(seq_len.item())
        
        # Predict
        with torch.no_grad():
            prediction = model(metadata_input, valid_len, None, 0.0)
        
        pred_np = prediction[0].cpu().numpy()
        gt_np = trajectory[:valid_len].cpu().numpy()
        
        # Plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Ground truth
        ax.plot(gt_np[:, 0], gt_np[:, 1], gt_np[:, 2],
                'g-', linewidth=3, label='Ground Truth', alpha=0.8)
        
        # Prediction
        ax.plot(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2],
                'r--', linewidth=2.5, label='Prediction', alpha=0.8)
        
        # Markers
        ax.plot([0], [0], [0], 'b*', markersize=20, label='Start', zorder=5)
        ax.plot([metadata[3].item()], [metadata[4].item()], [metadata[5].item()],
                'm*', markersize=20, label='Goal', zorder=5)
        
        ax.set_xlabel('East (m)', fontsize=13, fontweight='bold')
        ax.set_ylabel('North (m)', fontsize=13, fontweight='bold')
        ax.set_zlabel('Altitude (m)', fontsize=13, fontweight='bold')
        
        heading_deg = metadata[6].item() * 180 / np.pi
        climb_deg = metadata[7].item() * 180 / np.pi
        ax.set_title(f'Test Trajectory {i+1}\n'
                     f'Heading: {heading_deg:.0f}° | Climb: {climb_deg:.0f}°',
                     fontsize=15, fontweight='bold')
        
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Calculate MSE
        mse = np.mean((pred_np - gt_np)**2)
        max_error = np.max(np.linalg.norm(pred_np - gt_np, axis=1))
        
        textstr = f'MSE: {mse:.2f}\nMax Error: {max_error:.2f}m'
        ax.text2D(0.05, 0.95, textstr,
                  transform=ax.transAxes, fontsize=11,
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.savefig(f'{save_dir}/test_trajectory_{i+1}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Plot {i+1}/{num_samples} saved (MSE: {mse:.2f})")
    
    print(f"\n✓ All plots saved to: {save_dir}/")
    print("="*60)

# ===============================
# Main Execution
# ===============================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("RBE 577 HW3: DUBINS AIRPLANE TRAJECTORY PREDICTION")
    print("LSTM Encoder-Decoder Model")
    print("="*60)
    
    # Load or generate dataset
    if os.path.exists('Data.mat'):
        dataset = load_dataset_from_mat('Data.mat')
    else:
        dataset = generate_dataset()
    
    # Train model
    model, test_dataset = train_model(dataset)
    
    # Save final model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    torch.save(model.state_dict(), 'dubins_lstm_final.pth')
    print("✓ Final model saved: dubins_lstm_final.pth")
    print("✓ Best model saved: best_dubins_lstm.pth")
    print("="*60)
    
    # Generate prediction plots
    plot_predictions(model, test_dataset, num_samples=10)
    
    # Final summary
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  • best_dubins_lstm.pth - Best model checkpoint")
    print("  • dubins_lstm_final.pth - Final model")
    print("  • training_curves.png - Loss curves")
    print("  • plots/test_predictions/*.png - 10 test predictions")
    print("  • runs/ - TensorBoard logs")
    print("\nTo view TensorBoard:")
    print("  tensorboard --logdir=runs --port=6006")
    print("  Open: http://localhost:6006")
    print("="*60 + "\n")