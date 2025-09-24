"""
test_model.py - Test trained control allocation model on individual samples

Usage:
    python test_model.py

Make sure you have:
- best_model.pth (saved model weights)
- scalers.pkl (saved StandardScaler objects)
- The original training script in the same directory for imports
"""

import torch
import numpy as np
import pickle

# Import your model classes and functions from the main script
from HW1 import Autoencoder, GenerateBmatrix, ComputeTau

class ModelTester:
    def __init__(self, model_path='best_model.pth', scaler_path='scalers.pkl'):
        """
        Initialize the model tester
        
        Args:
            model_path: Path to saved model weights
            scaler_path: Path to saved StandardScaler objects
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = Autoencoder(hidden_size=64).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")
        
        # Load scalers
        try:
            with open(scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)
                self.input_scaler = scaler_data['input_scaler']
                self.output_scaler = scaler_data['output_scaler']
            print("Scaling info loaded")
        except FileNotFoundError:
            raise FileNotFoundError(f"Scaler file {scaler_path} not found. Make sure to run training first.")
        
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
        
        # Prepare input using StandardScaler
        tau_array = np.array([tau_forces])  # Shape: [1, 3]
        tau_scaled = self.input_scaler.transform(tau_array)
        tau_tensor = torch.tensor(tau_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            encoded, decoded = self.model(tau_tensor)
            
            # Unscale predicted commands using StandardScaler
            commands_scaled = encoded.cpu().numpy()
            commands_unscaled = self.output_scaler.inverse_transform(commands_scaled)
            
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
            decoded_unscaled = self.input_scaler.inverse_transform(decoded_scaled)[0]
        
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
        physics_error = np.sum((reconstructed_tau_np - tau_forces)**2)
        autoencoder_error = np.sum((decoded_unscaled - tau_forces)**2)
        
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
        ([10, 0, 0], "Pure Surge"),
        ([0, 10, 0], "Pure Sway"), 
        ([0, 0, 5], "Pure Yaw"),
        ([1, 5, 2], "Combined Motion"),
        ([20, -10, -3], "Complex Maneuver"),
        ([-15, 8, 1], "Reverse Motion"),
        ([0, 0, 0], "Hold Position"),
        ([5, 2, 1], "High Force Request"),
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
    
    print(f"\nTesting completed!")

if __name__ == "__main__":
    main()