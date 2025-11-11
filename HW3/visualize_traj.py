# Filename : visualize_traj.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from dubinEHF3d import dubinEHF3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

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

def predict_trajectory(model, start_pos, goal_pos, mean, std, max_steps=200, stop_threshold=15.0):
    """
    Predict trajectory using the LSTM model in autoregressive manner.
    
    Args:
        model: Trained LSTM model
        start_pos: Starting position [x, y, z]
        goal_pos: Goal position [x, y, z] (the actual 3D endpoint)
        mean: Normalization mean
        std: Normalization std
        max_steps: Maximum number of prediction steps
        stop_threshold: Distance threshold to stop prediction
    """
    model.eval()
    
    # Start with initial position
    current_pos = torch.tensor(start_pos, dtype=dtype)
    goal = torch.tensor(goal_pos, dtype=dtype)
    
    # Ensure mean and std are on the same device
    mean = mean.to(device)
    std = std.to(device)
    current_pos = current_pos.to(device)
    goal = goal.to(device)
    
    trajectory = [current_pos.cpu().numpy()]
    
    with torch.no_grad():
        for step in range(max_steps):
            # Normalize current position AND goal (to match training)
            current_pos_norm = (current_pos - mean) / std
            goal_norm = (goal - mean) / std
            
            # Create input: [normalized_pos (3), normalized_goal (3)]
            model_input = torch.cat([current_pos_norm, goal_norm]).unsqueeze(0).unsqueeze(0)  # [1, 1, 6]
            length = torch.tensor([1])
            
            # Predict next position (normalized)
            next_pos_norm = model(model_input, length).squeeze(0).squeeze(0)  # [3]
            
            # Denormalize
            next_pos = (next_pos_norm * std) + mean
            
            trajectory.append(next_pos.cpu().numpy())
            
            # Stop when close to goal
            distance_to_goal = torch.norm(next_pos - goal)
            if distance_to_goal < stop_threshold:
                print(f"    Stopped at step {step+1}: reached goal (distance={distance_to_goal:.2f})")
                break
            
            current_pos = next_pos
    
    return np.array(trajectory)

def visualize_comparison(model, mean, std, num_examples=10, save_dir='trajectory_plots'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    turn_radius = 60
    path_step_size = 10

    test_cases = [
        (0, 300, 200, 15),
        (45, -250, 300, -20),
        (90, 400, -100, 25),
        (135, -300, -300, 0),
        (180, 500, 0, 30),
        (225, 0, 400, -15),
        (270, -400, 200, 10),
        (315, 350, 350, -25),
        (60, -200, -400, 20),
        (150, 450, -250, -10),
    ]

    for idx, (start_heading_deg, goal_x, goal_y, climb_angle_deg) in enumerate(test_cases[:num_examples]):
        print(f"\nGenerating plot {idx+1}/{num_examples}...")
        start_heading = start_heading_deg * np.pi / 180
        climb_angle = climb_angle_deg * np.pi / 180

        # Generate ground truth Dubins path
        gt_path, end_heading, num_points = dubinEHF3d(
            0, 0, 0, start_heading, goal_x, goal_y,
            turn_radius, path_step_size, climb_angle, idx
        )

        if gt_path is None or len(gt_path) == 0:
            print(f"  Skipping case {idx+1}: No valid ground truth path")
            continue

        # CRITICAL FIX: Use num_points to get the actual endpoint
        # dubinEHF3d returns a padded array
        actual_goal_3d = gt_path[num_points - 1]  # Use num_points, not -1
        
        print(f"  Start: [0, 0, 0]")
        print(f"  GT Endpoint: [{actual_goal_3d[0]:.1f}, {actual_goal_3d[1]:.1f}, {actual_goal_3d[2]:.1f}]")
        print(f"  Requested Goal: [{goal_x}, {goal_y}]")
        print(f"  GT Path length: {num_points} points")

        # Predict trajectory using the LSTM
        pred_path = predict_trajectory(
            model,
            start_pos=[0.0, 0.0, 0.0],
            goal_pos=[actual_goal_3d[0], actual_goal_3d[1], actual_goal_3d[2]],
            mean=mean,
            std=std,
            max_steps=min(num_points * 2, 300),
            stop_threshold=path_step_size * 1.5
        )

        print(f"  Predicted path length: {len(pred_path)} points")

        # Plotting
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot ground truth
        ax.plot(gt_path[:num_points, 0], gt_path[:num_points, 1], gt_path[:num_points, 2],
                'b-', linewidth=2, label='Ground Truth', alpha=0.7)

        # Plot prediction
        if pred_path is not None and len(pred_path) > 1:
            ax.plot(pred_path[:, 0], pred_path[:, 1], pred_path[:, 2],
                    'r--', linewidth=2, label='LSTM Prediction', alpha=0.7)
            ax.scatter([pred_path[-1, 0]], [pred_path[-1, 1]], [pred_path[-1, 2]],
                       c='red', s=100, marker='x', label='Predicted End', zorder=6, linewidths=3)
        else:
            print(f"  Warning: Predicted path is empty or too short for case {idx+1}")

        # Plot markers
        ax.scatter([0], [0], [0], c='green', s=150, marker='o', label='Start', zorder=5)
        ax.scatter([actual_goal_3d[0]], [actual_goal_3d[1]], [actual_goal_3d[2]], 
                   c='purple', s=150, marker='*', label='Goal', zorder=5)

        # Compute errors
        min_len = min(len(gt_path[:num_points]), len(pred_path))
        if min_len > 0:
            pos_error = np.linalg.norm(gt_path[:min_len, :] - pred_path[:min_len, :], axis=1)
            mean_error = np.mean(pos_error)
            max_error = np.max(pos_error)
            endpoint_error = np.linalg.norm(pred_path[-1] - actual_goal_3d)
        else:
            mean_error = float('inf')
            max_error = float('inf')
            endpoint_error = float('inf')

        print(f"  Mean error: {mean_error:.2f}, Max error: {max_error:.2f}, Endpoint error: {endpoint_error:.2f}")

        ax.set_xlabel('X (East)', fontsize=12)
        ax.set_ylabel('Y (North)', fontsize=12)
        ax.set_zlabel('Z (Altitude)', fontsize=12)
        ax.set_title(f'Test Case {idx+1}: θ={start_heading_deg}°, Goal=({goal_x},{goal_y}), γ={climb_angle_deg}°\n'
                     f'Mean Error: {mean_error:.2f}, Max Error: {max_error:.2f}, Endpoint Error: {endpoint_error:.2f}', 
                     fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set equal aspect ratio
        max_range = np.array([
            gt_path[:num_points, 0].max() - gt_path[:num_points, 0].min(),
            gt_path[:num_points, 1].max() - gt_path[:num_points, 1].min(),
            gt_path[:num_points, 2].max() - gt_path[:num_points, 2].min()
        ]).max() / 2.0

        mid_x = (gt_path[:num_points, 0].max() + gt_path[:num_points, 0].min()) * 0.5
        mid_y = (gt_path[:num_points, 1].max() + gt_path[:num_points, 1].min()) * 0.5
        mid_z = (gt_path[:num_points, 2].max() + gt_path[:num_points, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/trajectory_comparison_{idx+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_dir}/trajectory_comparison_{idx+1}.png")

if __name__ == "__main__":
    print("Loading trained model...")
    
    # Try to load the new balanced model first, fallback to old name
    try:
        checkpoint = torch.load("./best_dubins_lstm_balanced.pth", map_location=device)
        print("Loaded: best_dubins_lstm_balanced.pth")
    except FileNotFoundError:
        try:
            checkpoint = torch.load("./dubins_lstm_balanced_final.pth", map_location=device)
            print("Loaded: dubins_lstm_balanced_final.pth")
        except FileNotFoundError:
            checkpoint = torch.load("./dubins_lstm_constrained_final.pth", map_location=device)
            print("Loaded: dubins_lstm_constrained_final.pth")
    
    model = DubinsLSTM().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    mean = checkpoint['mean']
    std = checkpoint['std']
    
    print(f"Model loaded successfully!")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'unknown')}")
    print(f"Normalization - Mean: {mean}, Std: {std}")
    if 'loss_weights' in checkpoint:
        print(f"Loss weights used during training:")
        for key, val in checkpoint['loss_weights'].items():
            print(f"  {key}: {val}")
    
    print("\n" + "="*60)
    print("Generating trajectory comparison plots...")
    print("="*60)
    visualize_comparison(model, mean, std, num_examples=10)
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)