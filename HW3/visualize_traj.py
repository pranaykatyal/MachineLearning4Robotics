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

def predict_trajectory(model, start_pos, start_heading, goal_x, goal_y, climb_angle, 
                       turn_radius, path_step_size, mean, std, max_steps=200):
    """
    Predict trajectory using the LSTM model in autoregressive manner.
    """
    model.eval()
    
    # Start with initial position
    current_pos = torch.tensor([start_pos[0], start_pos[1], start_pos[2]], dtype=dtype)
    goal = torch.tensor([goal_x, goal_y, climb_angle], dtype=dtype)
    
    # Ensure mean and std are on the same device as current_pos
    mean = mean.to(current_pos.device)
    std = std.to(current_pos.device)
    
    trajectory = [current_pos.cpu().numpy()]
    
    with torch.no_grad():
        for step in range(max_steps):
            # Normalize current position
            current_pos_norm = (current_pos - mean) / std
            
            # Create input: [normalized_pos (3), goal (3)]
            model_input = torch.cat([current_pos_norm, goal]).unsqueeze(0).unsqueeze(0)  # [1, 1, 6]
            length = torch.tensor([1])
            
            # Predict next position (normalized)
            next_pos_norm = model(model_input.to(device), length).squeeze(0).squeeze(0)  # [3]
            
            # Denormalize
            next_pos = (next_pos_norm.cpu() * std) + mean
            
            trajectory.append(next_pos.numpy())
            
            # Stop when close to Dubins endpoint
            distance_to_goal = torch.norm(next_pos - goal)
            if distance_to_goal < path_step_size * 2:  # Close enough to goal
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
        print(f"Generating plot {idx+1}/{num_examples}...")
        start_heading = start_heading_deg * np.pi / 180
        climb_angle = climb_angle_deg * np.pi / 180

        gt_path, end_heading, num_points = dubinEHF3d(
            0, 0, 0, start_heading, goal_x, goal_y,
            turn_radius, path_step_size, climb_angle, idx
        )

        if gt_path is None or len(gt_path) == 0:
            print(f"  Skipping case {idx+1}: No valid ground truth path")
            continue

        # Use actual Dubins endpoint as goal for prediction
        actual_goal = gt_path[-1]
        goal_x_pred, goal_y_pred, climb_angle_pred = actual_goal[0], actual_goal[1], actual_goal[2]

        pred_path = predict_trajectory(
            model,
            start_pos=[0, 0, 0],
            start_heading=start_heading,
            goal_x=goal_x_pred,
            goal_y=goal_y_pred,
            climb_angle=climb_angle_pred,
            turn_radius=turn_radius,
            path_step_size=path_step_size,
            mean=mean,
            std=std,
            max_steps=min(num_points * 2, 300)
        )

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(gt_path[:num_points, 0], gt_path[:num_points, 1], gt_path[:num_points, 2],
                'b-', linewidth=2, label='Ground Truth', alpha=0.7)

        if pred_path is not None and len(pred_path) > 1:
            ax.plot(pred_path[:, 0], pred_path[:, 1], pred_path[:, 2],
                    'r--', linewidth=2, label='LSTM Prediction', alpha=0.7)
            ax.scatter([pred_path[-1, 0]], [pred_path[-1, 1]], [pred_path[-1, 2]],
                       c='red', s=80, marker='x', label='Predicted End', zorder=6)
        else:
            print(f"  Warning: Predicted path is empty or too short for case {idx+1}")

        ax.scatter([0], [0], [0], c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter([goal_x], [goal_y], [0], c='purple', s=100, marker='*', label='Goal', zorder=5)

        min_len = min(len(gt_path[:num_points]), len(pred_path))
        if min_len > 0:
            pos_error = np.linalg.norm(gt_path[:min_len, :] - pred_path[:min_len, :], axis=1)
            mean_error = np.mean(pos_error)
            max_error = np.max(pos_error)
        else:
            mean_error = float('inf')
            max_error = float('inf')

        ax.set_xlabel('X (East)', fontsize=12)
        ax.set_ylabel('Y (North)', fontsize=12)
        ax.set_zlabel('Z (Altitude)', fontsize=12)
        ax.set_title(f'Test Case {idx+1}: θ={start_heading_deg}°, Goal=({goal_x},{goal_y}), γ={climb_angle_deg}°\n'
                     f'Mean Error: {mean_error:.2f}, Max Error: {max_error:.2f}', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

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
        print(f"  Mean position error: {mean_error:.2f}, Max error: {max_error:.2f}")

if __name__ == "__main__":
    print("Loading trained model...")
    checkpoint = torch.load("./best_dubins_lstm.pth", map_location=device)
    model = DubinsLSTM().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    mean = checkpoint['mean']
    std = checkpoint['std']
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'unknown')}")
    print(f"Normalization - Mean: {mean}, Std: {std}")
    print("\nGenerating trajectory comparison plots...")
    visualize_comparison(model, mean, std, num_examples=10)
    print("\nVisualization complete!")