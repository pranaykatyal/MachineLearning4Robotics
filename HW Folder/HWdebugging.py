import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time

class ShipControlDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ShipControlAllocator(nn.Module):
    def __init__(self, input_dim=3, output_dim=5, hidden_dim=16, num_layers=2, sequence_length=10):
        super(ShipControlAllocator, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Simplified architecture
        self.encoder_lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.encoder_output = nn.Linear(hidden_dim, output_dim)
        
        # Decoder
        self.decoder_lstm1 = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)
        
        self.init_weights()
        
        # Normalized constraints
        self.max_limits = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        self.max_rates = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
        
        # Reasonable loss weights
        self.k = torch.tensor([1.0, 1.0, 0.1, 0.1, 0.01, 0.1])
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encoder
        x_enc, _ = self.encoder_lstm1(x)
        commands = self.encoder_output(x_enc[:, -1, :])
        
        # Decoder
        commands_expanded = commands.unsqueeze(1).repeat(1, self.sequence_length, 1)
        x_dec, _ = self.decoder_lstm1(commands_expanded)
        reconstructed = self.decoder_output(x_dec)
        
        return commands, reconstructed
    
    def loss_L1(self, reconstructed, original):
        return nn.MSELoss()(reconstructed, original)
    
    def loss_L2(self, commands):
        violations = torch.clamp(torch.abs(commands) - self.max_limits.to(commands.device), min=0)
        return torch.mean(violations)
    
    def loss_L3(self, commands):
        if commands.size(0) > 1:
            rates = torch.abs(commands[1:] - commands[:-1])
            violations = torch.clamp(rates - self.max_rates.to(commands.device), min=0)
            return torch.mean(violations)
        return torch.tensor(0.0).to(commands.device)
    
    def loss_L4(self, commands):
        forces = torch.abs(commands[:, [0, 1, 3]])
        power = torch.mean(torch.pow(forces + 1e-8, 1.5))
        return power
    
    def loss_L5(self, commands):
        angles = commands[:, [2, 4]]
        sector_violation = torch.clamp(torch.abs(angles) - 0.8, min=0)
        return torch.mean(sector_violation)
    
    def compute_total_loss(self, commands, reconstructed, original):
        l1 = self.loss_L1(reconstructed, original)
        l2 = self.loss_L2(commands)
        l3 = self.loss_L3(commands)
        l4 = self.loss_L4(commands)
        l5 = self.loss_L5(commands)
        
        individual_losses = (l1.item(), l2.item(), l3.item(), l4.item(), l5.item())
        
        total_loss = (self.k[0] * l1 + self.k[1] * l1 +
                     self.k[2] * l2 + self.k[3] * l3 + 
                     self.k[4] * l4 + self.k[5] * l5)
        
        return total_loss, individual_losses

class TrainingPlotter:
    def __init__(self):
        plt.ion()  # Interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.train_losses = []
        self.val_losses = []
        self.individual_losses = []
        self.epochs = []
        
    def update(self, epoch, train_loss, val_loss, individual_loss):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.individual_losses.append(individual_loss)
        
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot total losses
        self.ax1.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        self.ax1.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        self.ax1.set_ylabel('Total Loss')
        self.ax1.set_yscale('log')
        self.ax1.legend()
        self.ax1.grid(True)
        self.ax1.set_title(f'Training Progress - Epoch {epoch}')
        
        # Plot individual losses
        individual_losses = np.array(self.individual_losses)
        labels = ['L1 (Recon)', 'L2 (Magnitude)', 'L3 (Rate)', 'L4 (Power)', 'L5 (Sector)']
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i in range(5):
            self.ax2.plot(self.epochs, individual_losses[:, i], 
                         color=colors[i], label=labels[i], linewidth=2)
        
        self.ax2.set_ylabel('Individual Losses')
        self.ax2.set_yscale('log')
        self.ax2.legend()
        self.ax2.grid(True)
        self.ax2.set_xlabel('Epoch')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def close(self):
        plt.ioff()
        plt.show()

class ShipControlTrainer:
    def __init__(self, ship_params):
        self.ship_params = ship_params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Data normalization parameters
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        
    def normalize_data(self, X, y):
        if self.X_mean is None:
            self.X_mean = np.mean(X, axis=(0, 1))
            self.X_std = np.std(X, axis=(0, 1)) + 1e-8
            self.y_mean = np.mean(y, axis=0)
            self.y_std = np.std(y, axis=0) + 1e-8
        
        X_normalized = (X - self.X_mean) / self.X_std
        y_normalized = (y - self.y_mean) / self.y_std
        
        return X_normalized, y_normalized
    
    def generate_training_data(self, num_samples=5000, sequence_length=10):
        # Normalized ranges
        F1_range = [-1.0, 1.0]
        F2_range = [-0.5, 0.5]
        F3_range = [-0.5, 0.5]
        a_range = [-1.0, 1.0]
        
        commands = []
        current = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
        for i in range(num_samples):
            step = np.random.normal(0, 0.01, 5).astype(np.float64)
            step[0] *= 0.1
            step[1] *= 0.05
            step[2] *= 0.01
            step[3] *= 0.05
            step[4] *= 0.01
            
            current += step
            
            current[0] = np.clip(current[0], F1_range[0], F1_range[1])
            current[1] = np.clip(current[1], F2_range[0], F2_range[1])
            current[2] = np.clip(current[2], a_range[0], a_range[1])
            current[3] = np.clip(current[3], F3_range[0], F3_range[1])
            current[4] = np.clip(current[4], a_range[0], a_range[1])
            
            commands.append(current.copy())
        
        commands = np.array(commands, dtype=np.float32)
        
        # Calculate generalized forces
        generalized_forces = []
        for cmd in commands:
            F1, F2, a2, F3, a3 = cmd
            F1_denorm = F1 * 10000
            F2_denorm = F2 * 5000
            F3_denorm = F3 * 5000
            a2_denorm = a2 * np.pi
            a3_denorm = a3 * np.pi
            
            surge = F1_denorm + F2_denorm * np.cos(a2_denorm) + F3_denorm * np.cos(a3_denorm)
            sway = F2_denorm * np.sin(a2_denorm) + F3_denorm * np.sin(a3_denorm)
            yaw = 5 * F2_denorm * np.sin(a2_denorm) - 8 * F3_denorm * np.cos(a3_denorm)
            
            surge_norm = surge / 20000.0
            sway_norm = sway / 10000.0
            yaw_norm = yaw / 50000.0
            
            generalized_forces.append([surge_norm, sway_norm, yaw_norm])
        
        X = np.array(generalized_forces, dtype=np.float32)
        y = commands.astype(np.float32)
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
        
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)
        
        return X_seq, y_seq
    
    def train(self, epochs=50, batch_size=32, learning_rate=0.001):
        print("Generating training data...")
        X_train, y_train = self.generate_training_data(num_samples=5000, sequence_length=10)
        
        # Split into train/validation
        split_idx = int(0.8 * len(X_train))
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]
        X_train, y_train = X_train[:split_idx], y_train[:split_idx]
        
        print(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
        
        # Normalize data
        X_train_norm, y_train_norm = self.normalize_data(X_train, y_train)
        X_val_norm, y_val_norm = self.normalize_data(X_val, y_val)
        
        # Create datasets
        train_dataset = ShipControlDataset(X_train_norm, y_train_norm)
        val_dataset = ShipControlDataset(X_val_norm, y_val_norm)
        
        train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)
        
        # Initialize model and optimizer
        model = ShipControlAllocator().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Initialize plotting
        plotter = TrainingPlotter()
        
        print("Starting training...")
        best_val_loss = float('inf')
        
        try:
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0
                individual_losses_epoch = np.zeros(5)
                num_batches = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    
                    if len(batch_X.shape) == 2:
                        batch_size_val = batch_X.size(0)
                        if batch_X.size(1) == 30:  # 10*3
                            batch_X = batch_X.view(batch_size_val, 10, 3)
                        else:
                            continue
                    
                    optimizer.zero_grad()
                    
                    commands, reconstructed = model(batch_X)
                    loss, individual_losses = model.compute_total_loss(commands, reconstructed, batch_X)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    individual_losses_epoch += np.array(individual_losses)
                    num_batches += 1
                
                if num_batches == 0:
                    continue
                    
                avg_train_loss = train_loss / num_batches
                avg_individual_losses = individual_losses_epoch / num_batches
                
                # Validation phase
                model.eval()
                val_loss = 0
                num_val_batches = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        
                        if len(batch_X.shape) == 2:
                            batch_size_val = batch_X.size(0)
                            if batch_X.size(1) == 30:
                                batch_X = batch_X.view(batch_size_val, 10, 3)
                            else:
                                continue
                        
                        commands, reconstructed = model(batch_X)
                        loss, _ = model.compute_total_loss(commands, reconstructed, batch_X)
                        val_loss += loss.item()
                        num_val_batches += 1
                
                avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else avg_train_loss
                
                # Update learning rate
                scheduler.step(avg_val_loss)
                
                # Update plot
                plotter.update(epoch + 1, avg_train_loss, avg_val_loss, avg_individual_losses)
                
                # Print progress
                if epoch % 5 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, '
                          f'Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}')
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), 'best_model.pth')
            
            print("Training completed!")
            
        except KeyboardInterrupt:
            print("Training interrupted by user")
        
        finally:
            plotter.close()
        
        return model
    
    def allocate_control(self, model, force_request):
        model.eval()
        
        with torch.no_grad():
            # Normalize force request
            force_request_norm = (np.array(force_request) - self.X_mean) / self.X_std
            force_tensor = torch.FloatTensor(force_request_norm).to(self.device)
            
            # Create sequence
            force_sequence = force_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 10, 1)
            
            commands_norm, _ = model(force_sequence)
            commands = commands_norm.cpu().numpy()[0] * self.y_std + self.y_mean
            
            return commands

def main():
    # Ship parameters
    ship_params = {
        'mass': 1.07e6,
        'length': 28.9,
        'breadth': 9.6,
        'draft': 2.8
    }
    
    # Create trainer
    trainer = ShipControlTrainer(ship_params)
    
    # Train model
    model = trainer.train(epochs=50, batch_size=32, learning_rate=0.001)
    
    # Test allocation
    force_request = [1000.0/20000, 500.0/10000, 200.0/50000]  # Normalized
    commands = trainer.allocate_control(model, force_request)
    
    # Denormalize commands
    commands_denorm = commands * np.array([10000, 5000, np.pi, 5000, np.pi])
    
    print(f"\nForce request: {force_request}")
    print(f"Allocated commands (denormalized):")
    print(f"  T1 force: {commands_denorm[0]:.2f} N")
    print(f"  T2 force: {commands_denorm[1]:.2f} N, angle: {np.rad2deg(commands_denorm[2]):.2f} deg")
    print(f"  T3 force: {commands_denorm[3]:.2f} N, angle: {np.rad2deg(commands_denorm[4]):.2f} deg")
    
    # Save final model
    torch.save(model.state_dict(), 'ship_control_allocator_final.pth')
    print("\nModel saved!")

if __name__ == "__main__":
    main()