import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from dubinEHF3d import dubinEHF3d

# ===============================
# Configuration
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Hyperparameters
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2

# Dataset generation parameters
TURN_RADIUS = 50
PATH_STEP_SIZE = 10
HEADING_INCREMENT = 15  # degrees
CLIMB_ANGLE_RANGE = 30  # degrees
CLIMB_ANGLE_INCREMENT = 5  # degrees
GRID_MARGIN = 5
GRID_SIZE = 2 * TURN_RADIUS * GRID_MARGIN
GRID_RESOLUTION = 25  # Coarser to reduce dataset size

# Maximum sequence length for trajectories
MAX_SEQ_LENGTH = 150

# ===============================
# Enhanced Dataset Generation
# ===============================
def generate_dataset():
    """Generate dataset with metadata for each trajectory."""
    dataset = []
    data_count = 0
    
    print("Generating dataset...")
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
                        # Store trajectory with metadata
                        trajectory = path[:num_points]
                        metadata = np.array([
                            0, 0, 0,  # start_x, start_y, start_z
                            x_goal, y_goal,  # goal_x, goal_y
                            start_heading,  # initial heading
                            climb_angle  # climb angle
                        ])
                        dataset.append({
                            'trajectory': trajectory,
                            'metadata': metadata,
                            'num_points': num_points
                        })
    
    print(f"Generated {len(dataset)} valid trajectories")
    return dataset

# ===============================
# Model Definition: Encoder-Decoder LSTM
# ===============================
class DubinsLSTMEncoderDecoder(nn.Module):
    """
    Encoder-Decoder LSTM for Dubins trajectory prediction.
    
    Encoder: Processes initial conditions (start, goal, heading, climb angle)
    Decoder: Generates trajectory sequence autoregressively
    """
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=3, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Encoder: processes initial conditions
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU()
        )
        
        # Decoder LSTM: generates trajectory
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
        """Initialize hidden state from encoded initial conditions."""
        # Split encoded representation into h0 and c0
        hidden_size = self.hidden_dim * self.num_layers
        h0 = encoded[:, :hidden_size].reshape(batch_size, self.num_layers, self.hidden_dim)
        c0 = encoded[:, hidden_size:].reshape(batch_size, self.num_layers, self.hidden_dim)
        
        # Transpose to (num_layers, batch, hidden_dim)
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
        encoded = self.encoder_fc(metadata)
        
        # Initialize hidden state
        hidden = self.init_hidden(batch_size, encoded)
        
        # Start decoding from origin (start position)
        decoder_input = torch.zeros(batch_size, 1, self.output_dim).to(metadata.device)
        decoder_input[:, 0, :] = metadata[:, :3]  # Start position (x, y, z)
        
        outputs = []
        
        for t in range(target_seq_len):
            # LSTM step
            lstm_out, hidden = self.decoder_lstm(decoder_input, hidden)
            
            # Generate prediction
            prediction = self.fc_out(lstm_out)
            outputs.append(prediction)
            
            # Teacher forcing: use ground truth as next input
            if target_trajectory is not None and np.random.random() < teacher_forcing_ratio:
                decoder_input = target_trajectory[:, t:t+1, :]
            else:
                decoder_input = prediction
        
        # Stack outputs: (batch, seq_len, 3)
        return torch.cat(outputs, dim=1)

# ===============================
# Training Function with TensorBoard
# ===============================
def train_model(dataset, run_name=None):
    """Train the model with TensorBoard logging."""
    
    # Create TensorBoard writer
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/dubins_{run_name}"
    writer = SummaryWriter(log_dir)
    
    print(f"\nTensorBoard logging to: {log_dir}")
    print("Run: tensorboard --logdir=runs")
    
    # Prepare data
    trajectories = []
    metadatas = []
    seq_lengths = []
    
    for sample in dataset:
        traj = torch.tensor(sample['trajectory'], dtype=dtype)
        meta = torch.tensor(sample['metadata'], dtype=dtype)
        
        # Pad or truncate trajectory to MAX_SEQ_LENGTH
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
    
    # Split: 80% train, 10% val, 10% test
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    
    # Initialize model
    model = DubinsLSTMEncoderDecoder(
        input_dim=7,
        hidden_dim=HIDDEN_DIM,
        output_dim=3,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Log model architecture
    writer.add_text("Model/Architecture", str(model))
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    criterion = nn.MSELoss()
    
    # Log hyperparameters
    hparams = {
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'weight_decay': WEIGHT_DECAY,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'max_seq_length': MAX_SEQ_LENGTH
    }
    writer.add_hparams(hparams, {})
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    train_losses = []
    val_losses = []
    
    for epoch in range(NUM_EPOCHS):
        # ==================== Training ====================
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (metadata, trajectory, seq_len) in enumerate(train_loader):
            metadata = metadata.to(device)
            trajectory = trajectory.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with teacher forcing (decreases over time)
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            # Log to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ==================== Validation ====================
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for metadata, trajectory, seq_len in val_loader:
                metadata = metadata.to(device)
                trajectory = trajectory.to(device)
                
                max_len = int(seq_len.max().item())
                predictions = model(metadata, max_len, target_trajectory=None, teacher_forcing_ratio=0.0)
                
                # Calculate loss only on valid timesteps
                loss = 0
                for i in range(len(seq_len)):
                    valid_len = int(seq_len[i].item())
                    loss += criterion(predictions[i, :valid_len], trajectory[i, :valid_len])
                loss /= len(seq_len)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, './best_dubins_lstm.pth')
            print(f"  ★ New best model saved! Val Loss: {best_val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # ==================== Test Evaluation ====================
    print("\n" + "="*50)
    print("Evaluating on Test Set...")
    
    checkpoint = torch.load('./best_dubins_lstm.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loss = 0.0
    with torch.no_grad():
        for metadata, trajectory, seq_len in test_loader:
            metadata = metadata.to(device)
            trajectory = trajectory.to(device)
            
            max_len = int(seq_len.max().item())
            predictions = model(metadata, max_len, target_trajectory=None, teacher_forcing_ratio=0.0)
            
            loss = 0
            for i in range(len(seq_len)):
                valid_len = int(seq_len[i].item())
                loss += criterion(predictions[i, :valid_len], trajectory[i, :valid_len])
            loss /= len(seq_len)
            
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.6f}")
    
    writer.add_scalar('Loss/test', avg_test_loss, 0)
    writer.close()
    
    # Save loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print(f"\nLoss curves saved to: training_curves.png")
    
    return model, test_dataset

# ===============================
# Visualization
# ===============================
def plot_predictions(model, test_dataset, num_samples=10):
    """Plot predicted vs ground truth trajectories."""
    save_dir = "plots/test_predictions"
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    print(f"\nGenerating {num_samples} prediction plots...")
    
    for i in range(min(num_samples, len(test_dataset))):
        metadata, trajectory, seq_len = test_dataset[i]
        
        # Prepare input
        metadata = metadata.unsqueeze(0).to(device)
        valid_len = int(seq_len.item())
        
        # Predict
        with torch.no_grad():
            prediction = model(metadata, valid_len, target_trajectory=None, teacher_forcing_ratio=0.0)
        
        # Convert to numpy
        pred_np = prediction[0].cpu().numpy()
        gt_np = trajectory[:valid_len].cpu().numpy()
        
        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Ground truth
        ax.plot(gt_np[:, 0], gt_np[:, 1], gt_np[:, 2], 
                'g-', linewidth=2, label='Ground Truth', alpha=0.8)
        
        # Prediction
        ax.plot(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2], 
                'r--', linewidth=2, label='Prediction', alpha=0.8)
        
        # Start and goal markers
        ax.plot([0], [0], [0], 'b*', markersize=15, label='Start')
        ax.plot([metadata[0, 3].item()], [metadata[0, 4].item()], [0], 
                'm*', markersize=15, label='Goal')
        
        ax.set_xlabel('East (m)', fontsize=12)
        ax.set_ylabel('North (m)', fontsize=12)
        ax.set_zlabel('Altitude (m)', fontsize=12)
        ax.set_title(f'Test Trajectory {i+1}\n'
                     f'Heading: {metadata[0, 5].item()*180/np.pi:.1f}°, '
                     f'Climb: {metadata[0, 6].item()*180/np.pi:.1f}°',
                     fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True)
        
        # Calculate error
        mse = np.mean((pred_np - gt_np)**2)
        ax.text2D(0.05, 0.95, f'MSE: {mse:.2f}', 
                  transform=ax.transAxes, fontsize=10,
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.savefig(f'{save_dir}/test_trajectory_{i+1}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Plots saved to: {save_dir}/")

# ===============================
# Main Execution
# ===============================
if __name__ == "__main__":
    print("="*70)
    print("RBE 577 - Homework 3: Dubins Airplane Trajectory Prediction with LSTM")
    print("="*70)
    
    # Generate dataset
    print("\n[1/4] Generating dataset...")
    dataset = generate_dataset()
    print(f"  ✓ Generated {len(dataset)} trajectories")
    
    # Train model
    print("\n[2/4] Training model...")
    model, test_dataset = train_model(dataset)
    print("  ✓ Training complete")
    
    # Save final model
    print("\n[3/4] Saving model...")
    torch.save(model.state_dict(), './dubins_lstm_final.pth')
    print("  ✓ Model saved to: dubins_lstm_final.pth")
    
    # Generate plots
    print("\n[4/4] Generating prediction plots...")
    plot_predictions(model, test_dataset, num_samples=10)
    print("  ✓ Plots generated")
    
    print("\n" + "="*70)
    print("✓ All tasks completed successfully!")
    print("="*70)
    print("\nTo view TensorBoard logs, run:")
    print("  tensorboard --logdir=runs")
    print("="*70)