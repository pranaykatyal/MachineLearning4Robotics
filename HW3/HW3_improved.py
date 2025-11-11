# Filename : HW3_improved.py
import torch
from dubinEHF3d import dubinEHF3d
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

BATCH_SIZE = 64
LR = 2e-3  # Increased from 1e-3
WEIGHT_DECAY = 1e-4  # Reduced from 1e-3
NUM_EPOCHS = 30  # Increased from 15

turn_radius = 30
path_step_size = 10

heading_increment = 10
climb_angle_range = 30
climb_angle_incrmement = 5

grid_margin = 5
grid_size = 2 * turn_radius * grid_margin
grid_resolution = 20

# Dataset generation options
USE_RANDOM_GOALS = True  # Set to True for random goals, False for grid
NUM_RANDOM_SAMPLES = 50000  # Number of random samples if using random goals

# BALANCED loss weights - less aggressive constraints
LOSS_WEIGHTS = {
    'mse': 1.0,              # Base MSE loss for trajectory tracking
    'endpoint': 1.0,         # REDUCED from 5.0 - balance with trajectory
    'smoothness': 0.1,       # REDUCED from 0.5 - less strict
    'curvature': 0.1,        # INCREASED from 0.05 - enforce turn radius
    'path_length': 0.0       # DISABLED - was causing inefficient paths
}

# No physical obstacles - invalid regions are defined by turn radius violations
INVALID_REGIONS = []

# Curvature constraint (maximum allowed curvature = 1/turn_radius)
MAX_CURVATURE = 1.0 / turn_radius

def generate_invalid_turn_sequence(start_pos, goal_pos, num_steps=10):
    """
    Generate a straight-line path that violates turn radius constraints.
    This is what the model should learn to avoid.
    """
    path = []
    for i in range(num_steps):
        t = i / (num_steps - 1)
        pos = start_pos + t * (goal_pos - start_pos)
        path.append(pos)
    return np.array(path)

def generate_dataset():
    dataset = []
    invalid_dataset = []  # Paths with impossible turns
    data_count = 0
    min_seq_len = 5
    
    print("Generating valid Dubins paths...")
    
    # FIRST: Generate all parameter combinations
    all_configs = []
    
    if USE_RANDOM_GOALS:
        # RANDOM sampling mode - better diversity
        print(f"Using RANDOM goal sampling (n={NUM_RANDOM_SAMPLES})")
        np.random.seed(42)
        
        for _ in range(NUM_RANDOM_SAMPLES):
            # Random climb angle
            climb_angle = np.random.randint(-climb_angle_range, climb_angle_range)
            
            # Random start heading
            start_heading = np.random.randint(0, 360)
            
            # Random goal in polar coordinates for uniform distribution
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(turn_radius * 2, grid_size / 2)  # At least 2x turn radius away
            
            x_spot = int(distance * np.cos(angle))
            y_spot = int(distance * np.sin(angle))
            
            all_configs.append((climb_angle, start_heading, x_spot, y_spot))
    else:
        # GRID sampling mode - systematic coverage
        print("Using GRID goal sampling")
        for climb_angle in range(-climb_angle_range, climb_angle_range, climb_angle_incrmement):
            for start_heading in range(0, 360, heading_increment):
                for x_spot in range(-grid_size // 2, grid_size // 2, grid_resolution):
                    for y_spot in range(-grid_size // 2, grid_size // 2, grid_resolution):
                        all_configs.append((climb_angle, start_heading, x_spot, y_spot))
        
        # RANDOMIZE the grid order to prevent sequential bias
        np.random.seed(42)
        np.random.shuffle(all_configs)
    
    print(f"Total configurations to generate: {len(all_configs)}")
    
    # Generate valid Dubins paths in RANDOM order
    for climb_angle, start_heading, x_spot, y_spot in all_configs:
        data_count += 1
        path, end_heading, num_points = dubinEHF3d(
            0, 0, 0,
            start_heading * (np.pi / 180),
            x_spot, y_spot,
            turn_radius, path_step_size,
            climb_angle * (np.pi / 180), data_count
        )
        # Only keep paths with enough steps to reach the goal
        if path is None or len(path) < min_seq_len or num_points < min_seq_len:
            continue
        
        # CRITICAL FIX: Use num_points to get the actual endpoint, not path[-1]
        actual_goal = path[num_points - 1]
        # Only use the valid portion of the path
        valid_path = path[:num_points]
        dataset.append((valid_path, actual_goal, end_heading, num_points))
    
    print(f"Generated {len(dataset)} valid paths")
    
    # Generate INVALID paths - these are straight lines that would require impossible turns
    print("\nGenerating invalid turn scenarios (straight-line paths)...")
    
    invalid_count = 0
    target_invalid_ratio = 0.15  # 15% of dataset should be invalid examples
    num_invalid_samples = int(len(dataset) * target_invalid_ratio)
    
    np.random.seed(42)
    
    for _ in range(num_invalid_samples):
        # Random start and goal that would require tight turns
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(100, 400)
        
        goal_x = distance * np.cos(angle)
        goal_y = distance * np.sin(angle)
        
        # Random climb angle
        climb_angle = np.random.uniform(-30, 30) * (np.pi / 180)
        goal_z = distance * np.tan(climb_angle)
        
        # Generate straight-line path (violates turn constraints)
        start_pos = np.array([0.0, 0.0, 0.0])
        goal_pos = np.array([goal_x, goal_y, goal_z])
        
        # Create a path that goes in a straight line with small steps
        num_steps = max(5, int(distance / path_step_size))
        invalid_path = generate_invalid_turn_sequence(start_pos, goal_pos, num_steps)
        
        invalid_dataset.append((invalid_path, goal_pos, 0, num_steps))
        invalid_count += 1
    
    print(f"Generated {invalid_count} invalid straight-line paths")
    print(f"Total dataset: {len(dataset) + invalid_count} samples")
    print(f"  Valid: {len(dataset)} ({100*len(dataset)/(len(dataset)+invalid_count):.1f}%)")
    print(f"  Invalid: {invalid_count} ({100*invalid_count/(len(dataset)+invalid_count):.1f}%)")
    
    # Combine valid and invalid datasets
    combined_dataset = dataset + invalid_dataset
    
    # SHUFFLE the combined dataset to mix valid and invalid examples
    np.random.shuffle(combined_dataset)
    print("Dataset shuffled for random training order")
    
    return combined_dataset

def compute_normalization_params(paths):
    all_points = torch.cat(paths, dim=0)
    mean = all_points.mean(dim=0)
    std = all_points.std(dim=0) + 1e-8
    return mean, std

def normalize_paths(paths, mean, std):
    return [(p - mean) / std for p in paths]

def pad_with_last(seq_list):
    max_len = max(len(seq) for seq in seq_list)
    padded = []
    lengths = []
    for seq in seq_list:
        lengths.append(len(seq))
        if len(seq) == 0:
            continue
        pad_len = max_len - len(seq)
        last_val = seq[-1].unsqueeze(0).repeat(pad_len, 1)
        padded_seq = torch.cat([seq, last_val], dim=0)
        padded.append(padded_seq)
    return padded, lengths

def compute_endpoint_loss(predictions, inputs, lengths, mask, mean, std):
    """
    Compute loss for reaching the ACTUAL GOAL accurately.
    The goal is embedded in inputs[:, :, 3:6] (positions 3,4,5).
    We want the last prediction to match the goal, not just the last target.
    """
    batch_size = predictions.size(0)
    endpoint_loss = 0.0
    
    for i in range(batch_size):
        # Get the last valid prediction
        last_idx = min(lengths[i] - 1, predictions.size(1) - 1)
        pred_endpoint = predictions[i, last_idx]  # Normalized prediction
        
        # Extract the GOAL from the input (it's the same for all timesteps)
        # Input structure: [pos_norm (3), goal_norm (3)]
        goal_norm = inputs[i, 0, 3:6]  # Goal is constant, just take first timestep
        
        # Compute endpoint loss in normalized space
        endpoint_loss += torch.sum((pred_endpoint - goal_norm) ** 2)
    
    return endpoint_loss / batch_size

def compute_smoothness_loss(predictions, lengths, mask):
    """
    Compute path smoothness loss based on acceleration (second derivative).
    """
    velocity = predictions[:, 1:] - predictions[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    
    acc_mask = torch.arange(acceleration.size(1), device=predictions.device).unsqueeze(0) < (lengths.unsqueeze(1) - 2)
    acc_mask = acc_mask.unsqueeze(2).expand_as(acceleration)
    
    smoothness = (acceleration ** 2 * acc_mask.float()).sum()
    valid_points = acc_mask.float().sum() + 1e-8
    
    return smoothness / valid_points

def compute_curvature_loss(predictions, lengths, mask, max_curvature):
    """
    Compute curvature constraint violation.
    This penalizes paths that require turns tighter than the aircraft can perform.
    """
    # Compute velocity and acceleration
    velocity = predictions[:, 1:] - predictions[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    
    # Align dimensions
    v_for_curv = velocity[:, :-1]
    
    # Compute curvature: |v x a| / |v|^3
    cross_product = torch.cross(v_for_curv, acceleration, dim=2)
    cross_mag = torch.norm(cross_product, dim=2)
    vel_mag = torch.norm(v_for_curv, dim=2)
    
    curvature = cross_mag / (vel_mag ** 3 + 1e-8)
    
    # Create mask for valid curvature points
    curv_mask = torch.arange(curvature.size(1), device=predictions.device).unsqueeze(0) < (lengths.unsqueeze(1) - 2)
    
    # Penalize curvature exceeding maximum (too tight turns)
    curvature_violation = torch.relu(curvature - max_curvature)
    curvature_loss = (curvature_violation ** 2 * curv_mask.float()).sum()
    valid_points = curv_mask.float().sum() + 1e-8
    
    return curvature_loss / valid_points

def compute_path_length_loss(predictions, lengths, mask):
    """
    Compute path length (disabled for now).
    """
    segment_lengths = torch.norm(predictions[:, 1:] - predictions[:, :-1], dim=2)
    seg_mask = torch.arange(segment_lengths.size(1), device=predictions.device).unsqueeze(0) < (lengths.unsqueeze(1) - 1)
    total_length = (segment_lengths * seg_mask.float()).sum()
    num_paths = predictions.size(0)
    return total_length / num_paths

def compute_comprehensive_loss(predictions, targets, lengths, inputs, mean, std):
    """
    Compute comprehensive loss combining all constraints.
    Note: inputs parameter is now used for endpoint loss
    """
    # Ensure predictions and targets have the same sequence length
    min_seq_len = min(predictions.size(1), targets.size(1))
    predictions = predictions[:, :min_seq_len, :]
    targets = targets[:, :min_seq_len, :]
    
    # Clamp lengths
    lengths = torch.clamp(lengths, max=min_seq_len)
    
    # Recreate mask
    mask = torch.arange(min_seq_len, device=predictions.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(2).expand_as(predictions)
    
    # 1. Base MSE loss
    mse_loss_per_point = (predictions - targets) ** 2
    masked_mse = (mse_loss_per_point * mask.float()).sum()
    valid_points = mask.float().sum() + 1e-8
    mse_loss = masked_mse / valid_points
    
    # 2. Endpoint loss - NOW USES ACTUAL GOAL!
    endpoint_loss = compute_endpoint_loss(predictions, inputs, lengths, mask, mean, std)
    
    # 3. Smoothness loss
    smoothness_loss = compute_smoothness_loss(predictions, lengths, mask)
    
    # 4. Curvature constraint loss (LEARNS INVALID REGIONS)
    curvature_loss = compute_curvature_loss(predictions, lengths, mask, MAX_CURVATURE)
    
    # 5. Path length loss
    path_length_loss = compute_path_length_loss(predictions, lengths, mask)
    
    # Combine all losses with weights
    total_loss = (
        LOSS_WEIGHTS['mse'] * mse_loss +
        LOSS_WEIGHTS['endpoint'] * endpoint_loss +
        LOSS_WEIGHTS['smoothness'] * smoothness_loss +
        LOSS_WEIGHTS['curvature'] * curvature_loss +
        LOSS_WEIGHTS['path_length'] * path_length_loss
    )
    
    # Return total loss and individual components for logging
    loss_components = {
        'total': total_loss,
        'mse': mse_loss,
        'endpoint': endpoint_loss,
        'smoothness': smoothness_loss,
        'curvature': curvature_loss,
        'path_length': path_length_loss
    }
    
    return total_loss, loss_components

def train_lstm(dataset):
    train_inputs = []
    train_targets = []

    for path_tuple in dataset:
        path = torch.tensor(path_tuple[0], dtype=dtype)
        actual_endpoint = torch.tensor(path_tuple[1], dtype=dtype)
        
        if path.shape[0] < 2:
            continue
            
        # Create input: current position + endpoint
        endpoint_seq = actual_endpoint.unsqueeze(0).repeat(path.shape[0] - 1, 1)
        inputs = torch.cat((path[:-1], endpoint_seq), dim=1)
        targets = path[1:]
        
        train_inputs.append(inputs)
        train_targets.append(targets)

    print("Computing normalization parameters...")
    input_points = [inp[:, :3] for inp in train_inputs]
    target_points = train_targets
    mean, std = compute_normalization_params(input_points + target_points)

    print(f"Normalization Mean: {mean}")
    print(f"Normalization Std: {std}")

    # Normalize BOTH position and goal
    train_inputs = [torch.cat([(inp[:, :3] - mean) / std, (inp[:, 3:] - mean) / std], dim=1) for inp in train_inputs]
    train_targets = normalize_paths(train_targets, mean, std)

    train_inputs, input_lengths = pad_with_last(train_inputs)
    train_targets, target_lengths = pad_with_last(train_targets)

    all_inputs = torch.stack(train_inputs)
    all_targets = torch.stack(train_targets)
    lengths = torch.tensor(input_lengths)

    print(f"Total dataset size: {len(all_inputs)}")
    print(f"Min sequence length: {lengths.min()}, Max sequence length: {lengths.max()}")

    total_size = all_inputs.size(0)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

    class SequenceDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, targets, lengths):
            self.inputs = inputs
            self.targets = targets
            self.lengths = lengths
        def __len__(self):
            return len(self.inputs)
        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx], self.lengths[idx]

    full_dataset = SequenceDataset(all_inputs, all_targets, lengths)

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class DubinsLSTM(nn.Module):
        def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, output_dim=3, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x, lengths):
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
            out = self.fc(out)
            return out

    model = DubinsLSTM().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    loss_history = {key: [] for key in ['mse', 'endpoint', 'smoothness', 'curvature', 'path_length']}

    writer = SummaryWriter(log_dir="runs/dubins_lstm_balanced")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_components = {key: 0.0 for key in loss_history.keys()}

        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        for batch_num, (batch_in, batch_out, batch_len) in enumerate(train_loader, 1):
            batch_in, batch_out, batch_len = batch_in.to(device), batch_out.to(device), batch_len.to(device)
            optimizer.zero_grad()

            preds = model(batch_in, batch_len)
            
            # Pass batch_in so endpoint loss can access the goal
            loss, loss_components = compute_comprehensive_loss(preds, batch_out, batch_len, batch_in, mean, std)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            for key in epoch_components.keys():
                epoch_components[key] += loss_components[key].item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar('Loss/train_total', avg_train_loss, epoch)
        
        for key in epoch_components.keys():
            avg_component = epoch_components[key] / len(train_loader)
            writer.add_scalar(f'Loss/train_{key}', avg_component, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        val_components = {key: 0.0 for key in loss_history.keys()}
        
        with torch.no_grad():
            for val_in, val_out, val_len in val_loader:
                val_in, val_out, val_len = val_in.to(device), val_out.to(device), val_len.to(device)

                preds = model(val_in, val_len)
                
                loss, loss_components = compute_comprehensive_loss(preds, val_out, val_len, val_in, mean, std)

                val_loss += loss.item()
                for key in val_components.keys():
                    val_components[key] += loss_components[key].item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        writer.add_scalar('Loss/validation_total', avg_val_loss, epoch)
        
        for key in val_components.keys():
            avg_component = val_components[key] / len(val_loader)
            writer.add_scalar(f'Loss/val_{key}', avg_component, epoch)
            loss_history[key].append(avg_component)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}")
        print(f"  Val Components:")
        for key in val_components.keys():
            print(f"    {key}: {val_components[key]/len(val_loader):.6f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'mean': mean,
                'std': std,
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'loss_weights': LOSS_WEIGHTS
            }, "./best_dubins_lstm_balanced.pth")
            print(f"\n★ Epoch {epoch+1}: New best validation loss {best_val_loss:.6f}")

    writer.close()

    # Test evaluation
    print("\nFinal evaluation on test set")
    test_loss = 0.0
    test_components = {key: 0.0 for key in loss_history.keys()}
    
    with torch.no_grad():
        for test_in, test_out, test_len in test_loader:
            test_in, test_out, test_len = test_in.to(device), test_out.to(device), test_len.to(device)

            preds = model(test_in, test_len)

            loss, loss_components = compute_comprehensive_loss(preds, test_out, test_len, test_in, mean, std)

            test_loss += loss.item()
            for key in test_components.keys():
                test_components[key] += loss_components[key].item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nFinal Test Loss: {avg_test_loss:.6f}")
    print(f"Test Components:")
    for key in test_components.keys():
        print(f"  {key}: {test_components[key]/len(test_loader):.6f}")

    # Plot training curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training and Validation Loss Components')
    
    axes[0, 0].plot(train_losses, label='Training Total')
    axes[0, 0].plot(val_losses, label='Validation Total')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    component_names = ['mse', 'endpoint', 'smoothness', 'curvature', 'path_length']
    for idx, key in enumerate(component_names):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        axes[row, col].plot(loss_history[key])
        axes[row, col].set_xlabel('Epoch')
        axes[row, col].set_ylabel(f'{key.capitalize()} Loss')
        axes[row, col].set_title(f'{key.capitalize()} Loss')
        axes[row, col].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_balanced.png')
    plt.close()

    return model, test_dataset, mean, std

print("Generating dataset...")
dataset = generate_dataset()
print(f"Dataset generated with {len(dataset)} samples")

print("\nLoss Configuration:")
print("=" * 50)
for key, value in LOSS_WEIGHTS.items():
    print(f"  {key}: {value}")
print(f"Max Curvature: {MAX_CURVATURE:.6f} (min turn radius: {turn_radius})")
print("\nThe model will learn that straight-line paths are INVALID")
print("because they require impossible tight turns!")

print("\nStarting training...")
model, test_dataset, mean, std = train_lstm(dataset)
print("\nTraining completed!")

print("Saving model...")
torch.save({
    'model_state_dict': model.state_dict(),
    'mean': mean,
    'std': std,
    'loss_weights': LOSS_WEIGHTS
}, "./dubins_lstm_balanced_final.pth")
print("Model saved to 'dubins_lstm_balanced_final.pth'")