"""
test_model.py - Test trained control allocation model on individual samples

Usage:
    python test_model.py

Make sure you have:
- best_model.pth (saved model weights)
- scaling_info.pkl (saved scaling parameters)
- The original training script in the same directory for imports
"""

import torch
import numpy as np
import pickle

# Import your model classes and functions from the main script
# Make sure your main script is named HW1.py or change the import
from HW1 import Autoencoder, GenerateBmatrix, ComputeTau

class ModelTester:
    def __init__(self, model_path='best_model.pth'):
        """
        Initialize the model tester
        
        Args:
            model_path: Path to saved model weights
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = Autoencoder(hidden_size=64).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")
        
        # Load scaling info
        try:
            with open('scaling_info.pkl', 'rb') as f:
                self.scaling_info = pickle.load(f)
            print("Scaling info loaded")
        except FileNotFoundError:
            print("Warning: scaling_info.pkl not found, using default values")
            self.scaling_info = {
                'force_scale': 10000.0,
                'angle_scale': 180.0, 
                'moment_scale': 10000.0
            }
        
    def scale_input(self, tau_forces):
        """Scale input tau forces"""
        tau_array = np.array([tau_forces])  # Shape: [1, 3]
        
        # Manual scaling using the same approach as training
        force_scale = self.scaling_info['force_scale']
        moment_scale = self.scaling_info['moment_scale']
        
        tau_scaled = tau_array.copy()
        tau_scaled[:, 0] = tau_array[:, 0] / force_scale   # Surge force
        tau_scaled[:, 1] = tau_array[:, 1] / force_scale   # Sway force  
        tau_scaled[:, 2] = tau_array[:, 2] / moment_scale  # Yaw moment
        
        return tau_scaled
    
    def unscale_output(self, commands_scaled):
        """Unscale output commands"""
        force_scale = self.scaling_info['force_scale']
        angle_scale = self.scaling_info['angle_scale']
        
        commands_unscaled = commands_scaled.copy()
        commands_unscaled[:, 0] = commands_scaled[:, 0] * force_scale  # F1
        commands_unscaled[:, 1] = commands_scaled[:, 1] * force_scale  # F2
        commands_unscaled[:, 2] = commands_scaled[:, 2] * angle_scale  # Alpha2
        commands_unscaled[:, 3] = commands_scaled[:, 3] * force_scale  # F3
        commands_unscaled[:, 4] = commands_scaled[:, 4] * angle_scale  # Alpha3
        
        return commands_unscaled
    
    def unscale_input(self, tau_scaled):
        """Unscale input tau forces"""
        force_scale = self.scaling_info['force_scale']
        moment_scale = self.scaling_info['moment_scale']
        
        tau_unscaled = tau_scaled.copy()
        tau_unscaled[:, 0] = tau_scaled[:, 0] * force_scale   # Surge force
        tau_unscaled[:, 1] = tau_scaled[:, 1] * force_scale   # Sway force
        tau_unscaled[:, 2] = tau_scaled[:, 2] * moment_scale  # Yaw moment
        
        return tau_unscaled
        
    def test_single_sample(self, tau_forces, sample_name="Sample"):
        """
        Test the network on a single sample
        
        Args:
            tau_forces: [surge_force, sway_force, yaw_moment] in Newtons/Nm
            sample_name: Name for display purposes
        """
        print(f"\n{'='*60}")
        print(f"TESTING {sample_name}")
        print(f"{'='*60}")
        
        # Prepare input using new scaling
        tau_scaled = self.scale_input(tau_forces)
        tau_tensor = torch.tensor(tau_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            encoded, decoded = self.model(tau_tensor)
            
            # Unscale predicted commands
            commands_scaled = encoded.cpu().numpy()
            commands_unscaled = self.unscale_output(commands_scaled)
            
            F1, F2, Alpha2, F3, Alpha3 = commands_unscaled[0]
            
            # Physics verification
            Alpha2_tensor = torch.tensor([Alpha2], device=self.device)
            Alpha3_tensor = torch.tensor([Alpha3], device=self.device)
            F1_tensor = torch.tensor([F1], device=self.device)
            F2_tensor = torch.tensor([F2], device=self.device)
            F3_tensor = torch.tensor([F3], device=self.device)
            
            B = GenerateBmatrix(Alpha2_tensor, Alpha3_tensor, 1).to(self.device)
            reconstructed_tau = ComputeTau(F1_tensor, F2_tensor, F3_tensor, B)
            reconstructed_tau_np = reconstructed_tau.cpu().numpy()[0]
            
            # Autoencoder reconstruction
            decoded_scaled = decoded.cpu().numpy()
            decoded_unscaled = self.unscale_input(decoded_scaled)[0]
        
        # Display results
        print(f"\n--- INPUT (Desired Forces) ---")
        print(f"Surge Force:  {tau_forces[0]:8.2f} N")
        print(f"Sway Force:   {tau_forces[1]:8.2f} N") 
        print(f"Yaw Moment:   {tau_forces[2]:8.2f} Nm")
        
        print(f"\n--- PREDICTED THRUSTER COMMANDS ---")
        print(f"F1 (Bow Thruster):     {F1:8.2f} N")
        print(f"F2 (Stern Thruster 1): {F2:8.2f} N")
        print(f"F3 (Stern Thruster 2): {F3:8.2f} N")
        print(f"Alpha2 (Angle 1):      {Alpha2:8.2f}°")
        print(f"Alpha3 (Angle 2):      {Alpha3:8.2f}°")
        
        print(f"\n--- VERIFICATION ---")
        print(f"Desired:    [{tau_forces[0]:8.2f}, {tau_forces[1]:8.2f}, {tau_forces[2]:8.2f}]")
        print(f"Physics:    [{reconstructed_tau_np[0]:8.2f}, {reconstructed_tau_np[1]:8.2f}, {reconstructed_tau_np[2]:8.2f}]")
        print(f"Autoenc:    [{decoded_unscaled[0]:8.2f}, {decoded_unscaled[1]:8.2f}, {decoded_unscaled[2]:8.2f}]")
        
        # Calculate errors
        physics_error = np.abs(reconstructed_tau_np - tau_forces).mean()
        autoencoder_error = np.abs(decoded_unscaled - tau_forces).mean()
        
        print(f"\n--- ERRORS ---")
        print(f"Physics Error (L0):     {physics_error:.6f}")
        print(f"Autoencoder Error (L1): {autoencoder_error:.6f}")
        
        # Constraint checks
        print(f"\n--- CONSTRAINT CHECKS ---")
        print(f"F1 within limits [-10000, 10000]:   {-10000 <= F1 <= 10000}")
        print(f"F2 within limits [-5000, 5000]:     {-5000 <= F2 <= 5000}")
        print(f"F3 within limits [-5000, 5000]:     {-5000 <= F3 <= 5000}")
        print(f"Alpha2 within limits [-180°, 180°]: {-180 <= Alpha2 <= 180}")
        print(f"Alpha3 within limits [-180°, 180°]: {-180 <= Alpha3 <= 180}")
        
        # Forbidden sectors check
        forbidden_2 = (80 < Alpha2 < 100) or (-100 < Alpha2 < -80)
        forbidden_3 = (80 < Alpha3 < 100) or (-100 < Alpha3 < -80)
        print(f"Alpha2 in forbidden sectors:         {forbidden_2}")
        print(f"Alpha3 in forbidden sectors:         {forbidden_3}")
        
        return {
            'commands': [F1, F2, F3, Alpha2, Alpha3],
            'physics_error': physics_error,
            'autoencoder_error': autoencoder_error,
            'original_tau': tau_forces,
            'reconstructed_tau': reconstructed_tau_np,
            'decoded_tau': decoded_unscaled
        }

def main():
    """
    Main testing function with predefined test cases
    """
    # Initialize tester
    tester = ModelTester()
    
    # Test cases: [surge, sway, yaw_moment]
    test_cases = [
        ([1000, 0, 0], "Pure Surge"),
        ([0, 1000, 0], "Pure Sway"), 
        ([0, 0, 500], "Pure Yaw"),
        ([1000, 500, 200], "Combined Motion"),
        ([2000, -1000, -300], "Complex Maneuver"),
        ([-1500, 800, 100], "Reverse Motion"),
        ([0, 0, 0], "Hold Position"),
        ([5000, 2000, 1000], "High Force Request"),
    ]
    
    print("CONTROL ALLOCATION NEURAL NETWORK TESTING")
    print("="*80)
    
    results = []
    for tau_forces, description in test_cases:
        result = tester.test_single_sample(tau_forces, description)
        results.append(result)
    
    # Summary statistics
    physics_errors = [r['physics_error'] for r in results]
    autoencoder_errors = [r['autoencoder_error'] for r in results]
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Average Physics Error:     {np.mean(physics_errors):.6f}")
    print(f"Average Autoencoder Error: {np.mean(autoencoder_errors):.6f}")
    print(f"Max Physics Error:         {np.max(physics_errors):.6f}")
    print(f"Max Autoencoder Error:     {np.max(autoencoder_errors):.6f}")
    print(f"Min Physics Error:         {np.min(physics_errors):.6f}")
    print(f"Min Autoencoder Error:     {np.min(autoencoder_errors):.6f}")
    
    # Test custom input
    print(f"\n{'='*80}")
    print("CUSTOM INPUT TEST")
    print(f"{'='*80}")
    print("Enter custom forces (or press Enter to skip):")
    
    try:
        surge = input("Surge force (N): ")
        if surge:
            sway = input("Sway force (N): ")
            yaw = input("Yaw moment (Nm): ")
            
            custom_forces = [float(surge), float(sway), float(yaw)]
            tester.test_single_sample(custom_forces, "Custom Input")
    except (ValueError, KeyboardInterrupt):
        print("Skipping custom input")
    
    print(f"\nTesting completed!")

if __name__ == "__main__":
    main()