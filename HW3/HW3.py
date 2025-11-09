import torch
from dubinEHF3d import dubinEHF3d
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import os

# ===============================
# Configuration
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-10  
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
# Training Function
# ===============================
def train_lstm(dataset):
    train_inputs = []
    train_targets = []

    for path_tuple in dataset:
        path = torch.tensor(path_tuple[0], dtype=dtype)
        inputs = path[:-1]
        targets = path[1:]
        train_inputs.append(inputs)
        train_targets.append(targets)

    # Pad sequences to make them same length
    all_inputs = pad_sequence(train_inputs, batch_first=True)
    all_targets = pad_sequence(train_targets, batch_first=True)
    full_dataset = TensorDataset(all_inputs, all_targets)

    # Split dataset: 80% train, 10% val, 10% test
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ===============================
    # Model Definition (LSTM)
    # ===============================
    class DubinsLSTM(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=64, num_layers=5, output_dim=3, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out)
            return out

    model = DubinsLSTM().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # ===============================
    # Training Loop
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
                    sample_pred = preds[0].cpu().numpy()
                    sample_target = batch_out[0].cpu().numpy()
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "./best_dubins_lstm.pth")
            print(f"\n★ Epoch {epoch+1}: New best validation loss {best_val_loss:.6f}")

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  Train Loss: {avg_train_loss:.6f}  Val Loss: {avg_val_loss:.6f}")

    # ===============================
    # Final Test Evaluation
    # ===============================
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

    return model, test_dataset


# ===============================
# Main Execution
# ===============================
print("Generating dataset...")
dataset = generate_dataset()
print(f"Dataset generated with {len(dataset)} samples")

print("\nStarting training...")
model, test_dataset = train_lstm(dataset)
print("\nTraining completed!")

print("Saving model...")
torch.save(model.state_dict(), "./dubins_lstm_final.pth")
print("Model saved to 'dubins_lstm_final.pth'")


# ===============================
# Prediction Function
# ===============================
def predict_trajectory(model, start_pos, num_steps=50):
    model.eval()
    trajectory = [torch.tensor(start_pos, dtype=dtype).to(device)]
    with torch.no_grad():
        pos = trajectory[0]
        for _ in range(num_steps):
            inp = pos.unsqueeze(0).unsqueeze(0)
            next_pos = model(inp)
            pos = next_pos.squeeze(0).squeeze(0)
            trajectory.append(pos)
    return torch.stack(trajectory).cpu().numpy()


# ===============================
# Plot Test Set Paths with Ground Truth
# ===============================
save_dir = "plots/test_paths"
os.makedirs(save_dir, exist_ok=True)

num_samples_to_plot = 10
subset_indices = range(min(num_samples_to_plot, len(test_dataset)))

print(f"\nSaving {len(subset_indices)} test paths with predictions and ground truth to '{save_dir}'...")

for i, idx in enumerate(subset_indices, start=1):
    inputs, targets = test_dataset[idx]

    inputs_np = inputs.cpu().numpy()
    targets_np = targets.cpu().numpy()

    num_valid_points = (targets_np != 0).any(axis=1).sum()
    if num_valid_points == 0:
        continue 
    inputs_np = inputs_np[:num_valid_points]
    targets_np = targets_np[:num_valid_points]

    model.eval()
    with torch.no_grad():
        inp_tensor = torch.tensor(inputs_np, dtype=dtype).unsqueeze(0).to(device)
        pred_np = model(inp_tensor).squeeze(0).cpu().numpy()
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

    path_file = os.path.join(save_dir, f"test_path_{i}.png")
    plt.savefig(path_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

print("✅ Test paths with ground truth and predictions saved successfully.")
