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
LR = 1e-3
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 15

turn_radius = 60
path_step_size = 10

heading_increment = 10
climb_angle_range = 30
climb_angle_incrmement = 5

grid_margin = 5
grid_size = 2 * turn_radius * grid_margin
grid_resolution = 30

def generate_dataset():
    dataset = []
    data_count = 0
    min_seq_len = 5  # Minimum sequence length required to reach the goal
    for climb_angle in range(-climb_angle_range, climb_angle_range, climb_angle_incrmement):
        for start_heading in range(0, 360, heading_increment):
            for x_spot in range(-grid_size // 2, grid_size // 2, grid_resolution):
                for y_spot in range(-grid_size // 2, grid_size // 2, grid_resolution):
                    data_count += 1
                    path, end_heading, num_points = dubinEHF3d(
                        0, 0, 0,
                        start_heading * (torch.pi / 180),
                        x_spot, y_spot,
                        turn_radius, path_step_size,
                        climb_angle * (torch.pi / 180), data_count
                    )
                    # Only keep paths with enough steps to reach the goal
                    if path is None or len(path) < min_seq_len:
                        continue
                    actual_goal = path[-1]
                    dataset.append((path, actual_goal, end_heading, num_points))
    return dataset

def compute_normalization_params(paths):
    all_points = torch.cat(paths, dim=0)
    mean = all_points.mean(dim=0)
    std = all_points.std(dim=0) + 1e-8
    return mean, std

def normalize_paths(paths, mean, std):
    return [(p - mean) / std for p in paths]

def denormalize_paths(paths, mean, std):
    return [(p * std) + mean for p in paths]

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

def train_lstm(dataset):
    train_inputs = []
    train_targets = []

    for path_tuple in dataset:
        path = torch.tensor(path_tuple[0], dtype=dtype)
        goal = torch.tensor(path_tuple[1], dtype=dtype)  # Now actual endpoint
        if path.shape[0] < 2:
            continue
        goal_seq = goal.unsqueeze(0).repeat(path.shape[0] - 1, 1)
        inputs = torch.cat((path[:-1], goal_seq), dim=1)
        targets = path[1:]
        train_inputs.append(inputs)
        train_targets.append(targets)

    print("Computing normalization parameters...")
    input_points = [inp[:, :3] for inp in train_inputs]
    target_points = train_targets
    mean, std = compute_normalization_params(input_points + target_points)

    print(f"Normalization Mean: {mean}")
    print(f"Normalization Std: {std}")

    # Normalize only the XYZ coordinates, keep goal as-is
    train_inputs = [torch.cat([(inp[:, :3] - mean) / std, inp[:, 3:]], dim=1) for inp in train_inputs]
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
    criterion = nn.MSELoss(reduction='none')

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    writer = SummaryWriter(log_dir="runs/dubins_lstm_fixed")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        total_valid_points = 0

        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        for batch_num, (batch_in, batch_out, batch_len) in enumerate(train_loader, 1):
            batch_in, batch_out, batch_len = batch_in.to(device), batch_out.to(device), batch_len.to(device)
            optimizer.zero_grad()

            preds = model(batch_in, batch_len)

            # Create mask for valid sequence positions
            mask = torch.arange(preds.size(1), device=device).unsqueeze(0) < batch_len.unsqueeze(1)
            mask = mask.unsqueeze(2).expand_as(preds)
            
            # Compute loss only on valid positions
            loss_per_point = criterion(preds, batch_out)
            masked_loss = (loss_per_point * mask.float()).sum()
            valid_points = mask.float().sum()
            
            loss = masked_loss / valid_points

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += masked_loss.item()
            total_valid_points += valid_points.item()

        avg_train_loss = epoch_loss / total_valid_points
        train_losses.append(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        total_val_points = 0
        
        with torch.no_grad():
            for val_in, val_out, val_len in val_loader:
                val_in, val_out, val_len = val_in.to(device), val_out.to(device), val_len.to(device)

                # FIXED: Data is already normalized, don't modify it!
                preds = model(val_in, val_len)

                mask = torch.arange(preds.size(1), device=device).unsqueeze(0) < val_len.unsqueeze(1)
                mask = mask.unsqueeze(2).expand_as(preds)
                
                loss_per_point = criterion(preds, val_out)
                masked_loss = (loss_per_point * mask.float()).sum()
                valid_points = mask.float().sum()

                val_loss += masked_loss.item()
                total_val_points += valid_points.item()

        avg_val_loss = val_loss / total_val_points
        val_losses.append(avg_val_loss)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'mean': mean,
                'std': std,
                'epoch': epoch,
                'val_loss': avg_val_loss
            }, "./best_dubins_lstm.pth")
            print(f"\n★ Epoch {epoch+1}: New best validation loss {best_val_loss:.6f}")

    writer.close()

    # Test evaluation
    print("\nFinal evaluation on test set")
    test_loss = 0.0
    total_test_points = 0
    
    with torch.no_grad():
        for test_in, test_out, test_len in test_loader:
            test_in, test_out, test_len = test_in.to(device), test_out.to(device), test_len.to(device)

            preds = model(test_in, test_len)

            mask = torch.arange(preds.size(1), device=device).unsqueeze(0) < test_len.unsqueeze(1)
            mask = mask.unsqueeze(2).expand_as(preds)

            loss_per_point = criterion(preds, test_out)
            masked_loss = (loss_per_point * mask.float()).sum()
            valid_points = mask.float().sum()

            test_loss += masked_loss.item()
            total_test_points += valid_points.item()

    avg_test_loss = test_loss / total_test_points
    print(f"\nFinal Test Loss: {avg_test_loss:.6f}")

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.close()

    return model, test_dataset, mean, std

print("Generating dataset...")
dataset = generate_dataset()
print(f"Dataset generated with {len(dataset)} samples")

print("\nStarting training...")
model, test_dataset, mean, std = train_lstm(dataset)
print("\nTraining completed!")

print("Saving model...")
torch.save({
    'model_state_dict': model.state_dict(),
    'mean': mean,
    'std': std
}, "./dubins_lstm_final.pth")
print("Model saved to 'dubins_lstm_final.pth'")