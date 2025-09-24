import importlib.util
import os
import pickle
import torch

HW1_PATH = os.path.join(os.path.dirname(__file__), 'HW1.py')
spec = importlib.util.spec_from_file_location('hw1mod', HW1_PATH)
hw1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hw1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# Try to find saved model and scalers
save_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
scaler_paths = [os.path.join(save_dir, 'scalers.pkl'), os.path.join(os.path.dirname(__file__), 'scalers.pkl')]
model_path = os.path.join(save_dir, 'autoencoder.pth')

input_scaler = None
output_scaler = None
if os.path.exists(scaler_paths[0]):
    with open(scaler_paths[0], 'rb') as f:
        d = pickle.load(f)
        input_scaler = d.get('input_scaler')
        output_scaler = d.get('output_scaler')
elif os.path.exists(scaler_paths[1]):
    with open(scaler_paths[1], 'rb') as f:
        d = pickle.load(f)
        input_scaler = d.get('input_scaler')
        output_scaler = d.get('output_scaler')

if not input_scaler or not output_scaler:
    print('No scalers found in', scaler_paths)
    raise SystemExit(1)

# Instantiate model and load weights if available
model = hw1.Autoencoder(hidden_size=64).to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print('Loaded model from', model_path)
else:
    print('No saved model at', model_path, '; using untrained model')

# Run the sanity test (it prints decoded unscaled tau)
hw1.sanity_zero_test(model, input_scaler, output_scaler, device)
