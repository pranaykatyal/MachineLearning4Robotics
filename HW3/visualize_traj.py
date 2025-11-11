import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from dubinEHF3d import dubinEHF3d
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def rotate_to_body_frame(points, heading):
    """Rotate points from world frame to body frame"""
    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)
    
    rotation_matrix = np.array([
        [cos_h, -sin_h, 0],
        [sin_h, cos_h, 0],
        [0, 0, 1]
    ])
    
    return points @ rotation_matrix.T

def rotate_from_body_frame(points, heading):
    """Rotate points from body frame back to world frame"""
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    
    rotation_matrix = np.array([
        [cos_h, -sin_h, 0],
        [sin_h, cos_h, 0],
        [0, 0, 1]
    ])
    
    return points @ rotation_matrix.T

class BodyFrameLSTM(nn.Module):
    """LSTM that predicts trajectory in body frame"""
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

def predict_trajectory(model, start_heading, goal_3d_world, climb_angle, 
                       turn_radius, path_step_size, mean, std, expected_length):
    """
    Predict trajectory using body-frame model.
    
    Args:
        goal_3d_world: [x, y, z] actual 3D endpoint in world frame
    """
    model.eval()
    
    # Convert goal to body frame - use ACTUAL 3D coordinates!
    goal_world = np.array([goal_3d_world])  # Shape: [1, 3]
    goal_body = rotate_to_body_frame(goal_world, start_heading)[0]  # Shape: [3]
    
    # Prepare input: [goal_body (3), climb_angle (1)]
    goal_body_tensor = torch.tensor(goal_body, dtype=dtype)
    goal_body_norm = (goal_body_tensor - mean) / std
    
    model_input = torch.cat([
        goal_body_norm,
        torch.tensor([climb_angle], dtype=dtype)
    ]).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Predict in body frame
        predictions_norm = model(model_input, expected_length).squeeze(0).cpu()
        
        # Denormalize
        predictions_body = (predictions_norm * std) + mean
        
        # Add start point (origin in body frame)
        start_body = torch.zeros(1, 3, dtype=dtype)
        full_traj_body = torch.cat([start_body, predictions_body], dim=0)
        
        # Convert back to world frame
        full_traj_world = rotate_from_body_frame(full_traj_body.numpy(), start_heading)
    
    return full_traj_world

def visualize_comparison(model, mean, std, num_examples=100, save_dir='trajectory_plots_body_frame'):
    """Generate comparison plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    turn_radius = 60
    path_step_size = 10

    # Match these to your training/test dataset ranges:
    heading_increment = 15
    climb_angle_range = 30
    grid_margin = 5
    grid_size = 2 * turn_radius * grid_margin
    grid_resolution = 40

    base_test_cases = [
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

    # If more examples requested, generate random test cases to fill up
    if num_examples > len(base_test_cases):
        np.random.seed(42)
        extra_cases = []
        for _ in range(num_examples - len(base_test_cases)):
            heading = np.random.choice(np.arange(0, 360, heading_increment))
            goal_x = np.random.choice(np.arange(-grid_size // 2, grid_size // 2, grid_resolution))
            goal_y = np.random.choice(np.arange(-grid_size // 2, grid_size // 2, grid_resolution))
            climb_angle = np.random.choice(np.arange(-climb_angle_range, climb_angle_range + 1, 5))
            extra_cases.append((heading, goal_x, goal_y, climb_angle))
        test_cases = base_test_cases + extra_cases
    else:
        test_cases = base_test_cases[:num_examples]

    errors = []
    
    for idx, (start_heading_deg, goal_x, goal_y, climb_angle_deg) in enumerate(test_cases[:num_examples]):
        print(f"Generating plot {idx+1}/{num_examples}...")
        
        start_heading = start_heading_deg * np.pi / 180
        climb_angle = climb_angle_deg * np.pi / 180
        
        # Generate ground truth
        gt_path, end_heading, num_points = dubinEHF3d(
            0, 0, 0, start_heading, goal_x, goal_y,
            turn_radius, path_step_size, climb_angle, idx
        )
        
        if gt_path is None or len(gt_path) == 0:
            print(f"  Skipping case {idx+1}: No valid ground truth path")
            continue
        
        # Use actual 3D endpoint
        actual_goal_3d = gt_path[num_points-1]
        
        # CRITICAL: Convert actual 3D goal to body frame (same as training!)
        # Predict trajectory
        pred_path = predict_trajectory(
            model,
            start_heading=start_heading,
            goal_3d_world=actual_goal_3d,  # Pass full 3D goal
            climb_angle=climb_angle,
            turn_radius=turn_radius,
            path_step_size=path_step_size,
            mean=mean,
            std=std,
            expected_length=num_points
        )
        
        # Calculate errors
        min_len = min(num_points, len(pred_path))
        pos_error = np.linalg.norm(gt_path[:min_len, :] - pred_path[:min_len, :], axis=1)
        mean_error = np.mean(pos_error)
        max_error = np.max(pos_error)
        final_error = np.linalg.norm(gt_path[num_points-1, :] - pred_path[-1, :])
        
        errors.append({
            'case': idx + 1,
            'mean': mean_error,
            'max': max_error,
            'final': final_error
        })
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectories
        ax.plot(gt_path[:num_points, 0], gt_path[:num_points, 1], gt_path[:num_points, 2],
                'b-', linewidth=2.5, label='Ground Truth', alpha=0.8)
        ax.plot(pred_path[:, 0], pred_path[:, 1], pred_path[:, 2],
                'r--', linewidth=2.5, label='LSTM Prediction (Body Frame)', alpha=0.8)
        
        # Mark points
        ax.scatter([0], [0], [0], c='green', s=150, marker='o', 
                   label='Start', zorder=5, edgecolors='black', linewidths=2)
        ax.scatter([actual_goal_3d[0]], [actual_goal_3d[1]], [actual_goal_3d[2]],
                   c='blue', s=150, marker='*', label='Goal / GT End', zorder=5, 
                   edgecolors='black', linewidths=2)
        ax.scatter([pred_path[-1, 0]], [pred_path[-1, 1]], [pred_path[-1, 2]],
                   c='red', s=120, marker='x', label='Pred End', zorder=6, linewidths=3)
        
        # Labels
        ax.set_xlabel('X (East) [m]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (North) [m]', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (Altitude) [m]', fontsize=12, fontweight='bold')
        
        title = (f'Test Case {idx+1}: θ₀={start_heading_deg}°, Goal=({goal_x},{goal_y}), γ={climb_angle_deg}°\n'
                 f'Mean Error: {mean_error:.2f}m, Max Error: {max_error:.2f}m, Final Error: {final_error:.2f}m')
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Equal aspect
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
        print(f"  Mean: {mean_error:.2f}m, Max: {max_error:.2f}m, Final: {final_error:.2f}m")
    
    if errors:
        print(f"\n{'='*60}")
        print("Summary Statistics:")
        print(f"{'='*60}")
        print(f"Average Mean Error:  {np.mean([e['mean'] for e in errors]):.2f}m")
        print(f"Average Max Error:   {np.mean([e['max'] for e in errors]):.2f}m")
        print(f"Average Final Error: {np.mean([e['final'] for e in errors]):.2f}m")
        print(f"{'='*60}")

if __name__ == "__main__":
    print("="*60)
    print("Loading trained body-frame model...")
    print("="*60)
    
    checkpoint = torch.load("./best_body_frame_lstm.pth", map_location=device)
    
    hyperparams = checkpoint.get('hyperparameters', {})
    hidden_dim = hyperparams.get('hidden_dim', 256)
    num_layers = hyperparams.get('num_layers', 3)
    dropout = hyperparams.get('dropout', 0.3)
    
    model = BodyFrameLSTM(
        input_dim=4,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=3,
        dropout=dropout
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    mean = checkpoint['mean']
    std = checkpoint['std']
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'unknown'):.6f}")
    print(f"Normalization - Mean: {mean}, Std: {std}")
    print(f"Architecture: hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}")
    
    print("\n" + "="*60)
    print("Generating trajectory comparison plots...")
    print("="*60 + "\n")
    
    visualize_comparison(model, mean, std, num_examples=100)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)