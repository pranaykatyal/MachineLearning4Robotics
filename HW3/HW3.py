
import torch
from dubinEHF3d import dubinEHF3d
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter

# ===============================
# Configuration
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-3  
NUM_EPOCHS = 20

turn_radius = 60
path_step_size = 10

heading_increment = 10
climb_angle_range = 30
climb_angle_incrmement = 5

grid_margin = 5
grid_size = 2 * turn_radius * grid_margin
grid_resolution = 10


# ===============================
# Dataset Generation
# ===============================
def generate_dataset():
    dataset = []
    data_count = 0
    for climb_angle in range(-climb_angle_range, climb_angle_range, climb_angle_incrmement): 
        for start_heading in range(0, 360, heading_increment):
            for x_spot in range(-grid_size//2, grid_size//2, grid_resolution):
                for y_spot in range(-grid_size//2, grid_size//2, grid_resolution):
                    data_count += 1
                    path, end_heading, num_points = dubinEHF3d(
                        0, 0, 0,
                        start_heading * (torch.pi / 180),
                        x_spot, y_spot,
                        turn_radius, path_step_size,
                        climb_angle * (torch.pi / 180), data_count
                    )
                    dataset.append((path, end_heading, num_points))
    return dataset


# ===============================
# Padding Strategy
# ===============================
def pad_with_last(seq_list):
    if len(seq_list) == 0:
        raise ValueError("pad_with_last received an empty sequence list.")
    max_len = max(len(s) for s in seq_list)
    padded = []
    for s in seq_list:
        pad_len = max_len - len(s)
        if pad_len > 0:
            pad_val = s[-1].unsqueeze(0).repeat(pad_len, 1)
            s = torch.cat([s, pad_val], dim=0)
        padded.append(s)
    return torch.stack(padded)


# ===============================
# Normalization Helpers
# ===============================
def compute_normalization_params(data_list):
    all_data = torch.cat(data_list, dim=0)
    mean = all_data.mean(dim=0)
    std = all_data.std(dim=0)
    std[std == 0] = 1e-8
    return mean, std

def normalize_tensor(t, mean, std):
    return (t - mean) / std

def denormalize_tensor(t, mean, std):
    return t * std + mean


# ===============================
# Training Function
# ===============================
def train_lstm(dataset):
    train_inputs, train_targets = [], []
    removed_count = 0

    for path_tuple in dataset:
        path = torch.tensor(path_tuple[0], dtype=dtype)
        if path.dim() == 0 or path.numel() == 0 or path.shape[0] < 2:
            removed_count += 1
            continue

        inputs = path[:-1]
        targets = path[1:]
        if inputs.shape[0] == 0 or targets.shape[0] == 0:
            removed_count += 1
            continue

        train_inputs.append(inputs)
        train_targets.append(targets)

    print(f"Filtered dataset: removed {removed_count} invalid/empty paths. Remaining sequences: {len(train_inputs)}")

    # Compute normalization params
    all_concat = train_inputs + train_targets
    mean, std = compute_normalization_params(all_concat)
    print(f"Dataset normalization → mean: {mean.tolist()}, std: {std.tolist()}")

    # Normalize
    norm_inputs = [normalize_tensor(x, mean, std) for x in train_inputs]
    norm_targets = [normalize_tensor(x, mean, std) for x in train_targets]

    # Pad normalized sequences
    all_inputs = pad_with_last(norm_inputs)
    all_targets = pad_with_last(norm_targets)
    full_dataset = TensorDataset(all_inputs, all_targets)

    # Split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ===============================
    # Model Definition
    # ===============================
    class DubinsLSTM(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=3, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                                batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out)
            return out

    model = DubinsLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # ===============================
    # Training Loop (original print style preserved)
    # ===============================
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        total_batches = len(train_loader)

        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        for batch_num, (batch_in, batch_out) in enumerate(train_loader, 1):
            batch_in, batch_out = batch_in.to(device), batch_out.to(device)
            optimizer.zero_grad()
            preds = model(batch_in)
            loss = criterion(preds, batch_out)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            running_loss = epoch_loss / batch_num

            if batch_num % max(1, total_batches // 10) == 0:
                print(f"\nBatch {batch_num}/{total_batches}")
                print(f"Current Batch Loss: {loss.item():.6f}")
                print(f"Running Avg Loss: {running_loss:.6f}")

                with torch.no_grad():
                    # Denormalize for display (exact original style)
                    sample_pred = denormalize_tensor(preds[0].cpu(), mean, std).numpy()
                    sample_target = denormalize_tensor(batch_out[0].cpu(), mean, std).numpy()
                    print("\nSample Prediction vs Target (first point in batch):")
                    print(f"Pred:   (x={sample_pred[0][0]:.2f}, y={sample_pred[0][1]:.2f}, z={sample_pred[0][2]:.2f})")
                    print(f"Target: (x={sample_target[0][0]:.2f}, y={sample_target[0][1]:.2f}, z={sample_target[0][2]:.2f})")

        avg_train_loss = epoch_loss / total_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_in, val_out in val_loader:
                val_in, val_out = val_in.to(device), val_out.to(device)
                preds = model(val_in)
                loss = criterion(preds, val_out)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # TensorBoard logging
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "./best_dubins_lstm.pth")
            print(f"\n★ Epoch {epoch+1}: New best validation loss {best_val_loss:.6f}")

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}")

    print("\nEvaluating on Test Set...")
    model.load_state_dict(torch.load("./best_dubins_lstm.pth"))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for test_in, test_out in test_loader:
            test_in, test_out = test_in.to(device), test_out.to(device)
            preds = model(test_in)
            loss = criterion(preds, test_out)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    print(f"\nFinal Test Loss: {avg_test_loss:.6f}")

    # Plot loss curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig("training_validation_loss.png", dpi=300, bbox_inches="tight")
    plt.show()

    return model, test_dataset, mean, std


# ===============================
# Main
# ===============================
print("Generating dataset...")
dataset = generate_dataset()
print(f"Dataset generated with {len(dataset)} samples")

print("\nStarting training...")
model, test_dataset, mean, std = train_lstm(dataset)
print("\nTraining completed!")

print("Saving model...")
torch.save(model.state_dict(), "./dubins_lstm_final.pth")
print("Model saved to 'dubins_lstm_final.pth'")


# ===============================
# Plot Predictions vs Ground Truth
# ===============================
save_dir = "plots/test_paths"
os.makedirs(save_dir, exist_ok=True)
num_samples_to_plot = 10
subset_indices = range(min(num_samples_to_plot, len(test_dataset)))

print(f"\nSaving {len(subset_indices)} test paths with predictions and ground truth to '{save_dir}'...")

for i, idx in enumerate(subset_indices, start=1):
    inputs, targets = test_dataset[idx]
    inputs_np = denormalize_tensor(inputs, mean, std).cpu().numpy()
    targets_np = denormalize_tensor(targets, mean, std).cpu().numpy()

    num_valid_points = (targets_np != 0).any(axis=1).sum()
    if num_valid_points == 0:
        continue
    inputs_np = inputs_np[:num_valid_points]
    targets_np = targets_np[:num_valid_points]

    model.eval()
    with torch.no_grad():
        inp_tensor = normalize_tensor(torch.tensor(inputs_np, dtype=dtype), mean, std).unsqueeze(0).to(device)
        pred_np = model(inp_tensor).squeeze(0).cpu()
        pred_np = denormalize_tensor(pred_np, mean, std).numpy()
        pred_np = pred_np[:len(targets_np)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(targets_np[:, 0], targets_np[:, 1], targets_np[:, 2], 'g-', label='Ground Truth')
    ax.plot(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2], 'r--', label='Model Prediction')
    ax.set_xlabel('East')
    ax.set_ylabel('North')
    ax.set_zlabel('Altitude')
    ax.set_title(f"Test Path {i}")
    ax.legend()

    plt.savefig(os.path.join(save_dir, f"test_path_{i}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

print("✅ Test paths with ground truth and predictions saved successfully.")
