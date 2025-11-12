import torch
from dubinEHF3d import dubinEHF3d
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Hyperparameters
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
HIDDEN_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.3

# Dubins parameters
turn_radius = 60
path_step_size = 10

# Data generation parameters
heading_increment = 15
climb_angle_range = 30
climb_angle_increment = 5
grid_margin = 5
grid_size = 2 * turn_radius * grid_margin
grid_resolution = 10

def rotate_to_body_frame(points, heading):
    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)
    rotation_matrix = np.array([
        [cos_h, -sin_h, 0],
        [sin_h, cos_h, 0],
        [0, 0, 1]
    ])
    return points @ rotation_matrix.T

def rotate_from_body_frame(points, heading):
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rotation_matrix = np.array([
        [cos_h, -sin_h, 0],
        [sin_h, cos_h, 0],
        [0, 0, 1]
    ])
    return points @ rotation_matrix.T

def generate_dataset():
    dataset = []
    data_count = 0
    min_seq_len = 5
    print("Generating dataset...")
    for climb_angle in range(-climb_angle_range, climb_angle_range + 1, climb_angle_increment):
        for start_heading in range(0, 360, heading_increment):
            for x_spot in range(-grid_size // 2, grid_size // 2, grid_resolution):
                for y_spot in range(-grid_size // 2, grid_size // 2, grid_resolution):
                    data_count += 1
                    heading_rad = start_heading * (np.pi / 180)
                    climb_rad = climb_angle * (np.pi / 180)
                    path, end_heading, num_points = dubinEHF3d(
                        0, 0, 0,
                        heading_rad,
                        x_spot, y_spot,
                        turn_radius, path_step_size,
                        climb_rad, 
                        data_count
                    )
                    if path is None or len(path) < min_seq_len:
                        continue
                    path_body = rotate_to_body_frame(path[:num_points], heading_rad)
                    dataset.append((
                        path_body,
                        heading_rad,
                        climb_rad,
                        num_points,
                        heading_rad
                    ))
    print(f"Generated {len(dataset)} valid trajectories")
    return dataset

class TrajectoryDataset(Dataset):
    def __init__(self, data_list, mean, std):
        self.data = data_list
        self.mean = mean.to(dtype)
        self.std = std.to(dtype)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        path_body, start_heading, climb_angle, num_points, _ = self.data[idx]
        path_tensor = torch.tensor(path_body, dtype=dtype)
        goal_body = path_tensor[-1]
        goal_body_norm = (goal_body - self.mean) / self.std
        input_vec = torch.cat([
            goal_body_norm,
            torch.tensor([climb_angle], dtype=dtype)
        ])
        target = (path_tensor[1:] - self.mean) / self.std
        return input_vec, target, len(target)

def collate_fn(batch):
    inputs, targets, lengths = zip(*batch)
    inputs = torch.stack(inputs)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    lengths = torch.tensor(lengths)
    return inputs, targets_padded, lengths

class BodyFrameLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, num_layers=3, output_dim=3, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    def forward(self, x, max_seq_len):
        encoded = self.encoder(x)
        lstm_input = encoded.unsqueeze(1).repeat(1, max_seq_len, 1)
        lstm_out, _ = self.lstm(lstm_input)
        output = self.fc(lstm_out)
        return output

def compute_normalization_params(dataset):
    all_points = []
    for path_body, _, _, num_points, _ in dataset:
        all_points.append(torch.tensor(path_body[:num_points], dtype=dtype))
    all_points = torch.cat(all_points, dim=0)
    mean = all_points.mean(dim=0)
    std = all_points.std(dim=0) + 1e-8
    return mean, std

def train_model(dataset):
    print("\nComputing normalization parameters...")
    mean, std = compute_normalization_params(dataset)
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    train_dataset = TrajectoryDataset([dataset[i] for i in train_data.indices], mean, std)
    val_dataset = TrajectoryDataset([dataset[i] for i in val_data.indices], mean, std)
    test_dataset = TrajectoryDataset([dataset[i] for i in test_data.indices], mean, std)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    model = BodyFrameLSTM(
        input_dim=4,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=3,
        dropout=DROPOUT
    ).to(device)
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    writer = SummaryWriter(log_dir="runs/dubins_body_frame")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        total_points = 0
        for batch_inputs, batch_targets, batch_lengths in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_lengths = batch_lengths.to(device)
            optimizer.zero_grad()
            max_len = batch_targets.size(1)
            predictions = model(batch_inputs, max_len)
            mask = torch.arange(max_len, device=device).unsqueeze(0) < batch_lengths.unsqueeze(1)
            mask = mask.unsqueeze(2).expand_as(predictions)
            loss_per_point = criterion(predictions, batch_targets)
            masked_loss = (loss_per_point * mask.float()).sum()
            valid_points = mask.float().sum()
            loss = masked_loss / valid_points
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += masked_loss.item()
            total_points += valid_points.item()
        avg_train_loss = epoch_loss / total_points
        train_losses.append(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        model.eval()
        val_loss = 0.0
        total_val_points = 0
        with torch.no_grad():
            for val_inputs, val_targets, val_lengths in val_loader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                val_lengths = val_lengths.to(device)
                max_len = val_targets.size(1)
                predictions = model(val_inputs, max_len)
                mask = torch.arange(max_len, device=device).unsqueeze(0) < val_lengths.unsqueeze(1)
                mask = mask.unsqueeze(2).expand_as(predictions)
                loss_per_point = criterion(predictions, val_targets)
                masked_loss = (loss_per_point * mask.float()).sum()
                valid_points = mask.float().sum()
                val_loss += masked_loss.item()
                total_val_points += valid_points.item()
        avg_val_loss = val_loss / total_val_points
        val_losses.append(avg_val_loss)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}]  "
              f"Train Loss: {avg_train_loss:.6f}  "
              f"Val Loss: {avg_val_loss:.6f}  "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'mean': mean,
                'std': std,
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'hyperparameters': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'dropout': DROPOUT,
                    'lr': LR,
                    'weight_decay': WEIGHT_DECAY
                }
            }, "./best_body_frame_lstm.pth")
            print(f"  ★ New best model saved! Val Loss: {best_val_loss:.6f}")
    writer.close()
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    total_test_points = 0
    with torch.no_grad():
        for test_inputs, test_targets, test_lengths in test_loader:
            test_inputs = test_inputs.to(device)
            test_targets = test_targets.to(device)
            test_lengths = test_lengths.to(device)
            max_len = test_targets.size(1)
            predictions = model(test_inputs, max_len)
            mask = torch.arange(max_len, device=device).unsqueeze(0) < test_lengths.unsqueeze(1)
            mask = mask.unsqueeze(2).expand_as(predictions)
            loss_per_point = criterion(predictions, test_targets)
            masked_loss = (loss_per_point * mask.float()).sum()
            valid_points = mask.float().sum()
            test_loss += masked_loss.item()
            total_test_points += valid_points.item()
    avg_test_loss = test_loss / total_test_points
    print(f"Final Test Loss: {avg_test_loss:.6f}")
    print(f"{'='*60}\n")
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training and Validation Loss (Body Frame)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves_body_frame.png', dpi=150)
    plt.close()
    print("Training curves saved to 'training_curves_body_frame.png'")
    return model, mean, std

if __name__ == "__main__":
    print("="*60)
    print("Dubins Trajectory Prediction - Body Frame Approach")
    print("="*60)
    dataset = generate_dataset()
    model, mean, std = train_model(dataset)
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)