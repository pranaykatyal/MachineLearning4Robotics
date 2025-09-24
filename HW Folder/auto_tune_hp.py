import importlib.util
import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import time

HW1_PATH = os.path.join(os.path.dirname(__file__), 'HW1.py')
spec = importlib.util.spec_from_file_location('hw1mod', HW1_PATH)
hw1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hw1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Grid to try (moderate size)
LRS = [5e-5, 1e-4]
K0S = [1.0, 2.0]
ZWS = [50.0, 200.0]
MODES = ['flat', 'sequence']

# Early stop target per-loss
EPS_TARGET = 1e-2

RESULTS = []

SCALER_MAP = {
    'standard': StandardScaler,
    'robust': RobustScaler,
    'minmax': MinMaxScaler,
}


def train_trial(lr, k0_final, zero_start, mode, trial_name, scaler_name='standard', epochs=150, num_sequences=32, seq_len=8, clip_tau2=None, signed_log_tau2=False):
    torch.manual_seed(42); np.random.seed(42)
    start_time = time.time()
    # Generate dataset as sequences
    F1, F2, F3, Alpha2, Alpha3, Tau = hw1.GenerateLargeDataset(num_sequences=num_sequences, sequence_length=seq_len)
    thruster_commands = torch.stack([F1, F2, Alpha2, F3, Alpha3], dim=1)

    inputs = Tau.numpy(); outputs = thruster_commands.numpy()
    # optional clipping on the third tau channel (index 2) to tame heavy tails
    if clip_tau2 is not None:
        try:
            inputs[:, 2] = np.clip(inputs[:, 2], -float(clip_tau2), float(clip_tau2))
        except Exception:
            # defensive: if shape unexpected, ignore clipping
            pass
    # optional signed-log transform on tau[2] to compress heavy tails while preserving sign
    inputs_for_scaling = inputs.copy()
    if signed_log_tau2:
        try:
            # y = sign(x) * log1p(|x|)
            inputs_for_scaling[:, 2] = np.sign(inputs_for_scaling[:, 2]) * np.log1p(np.abs(inputs_for_scaling[:, 2]))
        except Exception:
            signed_log_tau2 = False
    ScalerCls = SCALER_MAP.get(scaler_name, StandardScaler)
    input_scaler = ScalerCls(); output_scaler = StandardScaler()
    inputs_scaled = input_scaler.fit_transform(inputs_for_scaling)
    outputs_scaled = output_scaler.fit_transform(outputs)

    # reshape for sequence mode
    N = inputs_scaled.shape[0]
    if mode == 'sequence':
        # reshape to [num_sequences, seq_len, features]
        inputs_seq = inputs_scaled.reshape(num_sequences, seq_len, -1)
        outputs_seq = outputs_scaled.reshape(num_sequences, seq_len, -1)
        X = torch.tensor(inputs_seq, dtype=torch.float32).to(DEVICE)
        Y = torch.tensor(outputs_seq, dtype=torch.float32).to(DEVICE)
        dataset = torch.utils.data.TensorDataset(X, Y)
        # In sequence mode, we'll iterate over sequences (batch is sequences)
        batch_size = 8
    else:
        # flat mode: samples are timesteps
        X = torch.tensor(inputs_scaled, dtype=torch.float32).to(DEVICE)
        Y = torch.tensor(outputs_scaled, dtype=torch.float32).to(DEVICE)
        dataset = torch.utils.data.TensorDataset(X, Y)
        batch_size = 256

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = hw1.Autoencoder(hidden_size=128).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

    # curriculum parameters
    zero_start_val = zero_start
    zero_end = 1.0
    k0_start = 0.1
    k0_end = k0_final

    zero_scaled = input_scaler.transform(np.zeros((1,3), dtype=np.float32))
    zero_scaled_t = torch.tensor(zero_scaled, dtype=torch.float32).to(DEVICE)

    best_metric = float('inf')
    best_state = None

    for ep in range(epochs):
        alpha = ep / float(max(1, epochs-1))
        zero_weight = zero_start_val * (1 - alpha) + zero_end * alpha
        k0 = k0_start * (1 - alpha) + k0_end * alpha

        model.train()
        total_loss = 0.0
        losses_accum = []
        for batch in loader:
            if mode == 'sequence':
                xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
                # flatten sequence batch to timesteps for loss computations
                b, s, f = xb.shape
                xb_flat = xb.reshape(b*s, f)
                yb_flat = yb.reshape(b*s, -1)
                # forward: model accepts 3D as well, but we'll flatten for encoder which unsqueezes
                enc, dec = model(xb_flat)
            else:
                xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
                xb_flat = xb
                yb_flat = yb
                enc, dec = model(xb_flat)

            # compute losses (all operate on flattened tensors)
            L1 = hw1.ComputeL1Loss(dec, xb_flat)
            L0 = hw1.ComputeL0Loss(enc, xb_flat, output_scaler, input_scaler, DEVICE)
            L2 = hw1.ComputeL2Loss(enc, output_scaler, DEVICE)
            L3 = hw1.ComputeL3Loss(enc, output_scaler, DEVICE)
            L4 = hw1.ComputeL4Loss(enc)
            L5 = hw1.ComputeL5Loss(enc, output_scaler, DEVICE)

            # zero reg
            _, decz = model(zero_scaled_t)
            Lz = hw1.ComputeL1Loss(decz, zero_scaled_t)

            loss = k0*L0 + 1.0*L1 + 0.1*L2 + 1e-7*L3 + 1e-7*L4 + 0.1*L5 + zero_weight * Lz
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total_loss += loss.item()
            losses_accum.append((L0.item(), L1.item(), L2.item(), L3.item(), L4.item(), L5.item(), Lz.item()))
        scheduler.step()

        # compute mean losses for epoch
        mean_losses = np.mean(np.array(losses_accum), axis=0)
        L0m, L1m, L2m, L3m, L4m, L5m, Lzm = mean_losses.tolist()
        metric = L0m + L1m + L2m + L3m + L4m + L5m + Lzm
        if metric < best_metric:
            best_metric = metric
            best_state = model.state_dict()

        if (ep+1) % 10 == 0 or ep==0:
            elapsed = time.time() - start_time
            print(f"{trial_name} ep {ep+1}/{epochs} lr={scheduler.get_last_lr()[0]:.2e} k0={k0:.3f} z={zero_weight:.1f} losses L0={L0m:.4e} L1={L1m:.4e} L2={L2m:.4e} L3={L3m:.4e} L4={L4m:.4e} L5={L5m:.4e} Lz={Lzm:.4e} time={elapsed:.0f}s")

        # early stop if all per-loss below EPS_TARGET
        if L0m < EPS_TARGET and L1m < EPS_TARGET and L2m < EPS_TARGET and L3m < EPS_TARGET and L4m < EPS_TARGET and L5m < EPS_TARGET:
            print(f"{trial_name} reached eps target at epoch {ep+1}")
            break

    # evaluate decoded zero
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, decz = model(zero_scaled_t)
        dec_np = decz.cpu().numpy().reshape(-1)
        # Try standard inverse_transform first, fallback to common scaler attributes
        try:
            phys = input_scaler.inverse_transform(dec_np.reshape(1, -1))[0]
        except Exception:
            if hasattr(input_scaler, 'scale_') and hasattr(input_scaler, 'mean_'):
                phys = dec_np * input_scaler.scale_ + input_scaler.mean_
            elif hasattr(input_scaler, 'scale_') and hasattr(input_scaler, 'center_'):
                phys = dec_np * input_scaler.scale_ + input_scaler.center_
            elif hasattr(input_scaler, 'data_min_') and hasattr(input_scaler, 'data_max_'):
                fr = getattr(input_scaler, 'feature_range', (0, 1))
                data_min = input_scaler.data_min_
                data_max = input_scaler.data_max_
                scale = (data_max - data_min) / (fr[1] - fr[0])
                phys = (dec_np - fr[0]) * scale + data_min
            else:
                phys = dec_np
        # if we applied signed-log preprocessing to tau[2], invert it here: x = sign(y) * (exp(|y|)-1)
        if signed_log_tau2:
            try:
                v = phys[2]
                phys = phys.copy()
                phys[2] = np.sign(v) * (np.expm1(abs(v)))
            except Exception:
                pass
    duration = time.time() - start_time
    return {
        'trial': trial_name,
        'mode': mode,
        'lr': lr,
        'k0_final': k0_final,
        'zero_start': zero_start,
        'clip_tau2': clip_tau2,
        'signed_log_tau2': signed_log_tau2,
        'scaler': scaler_name,
        'best_metric': best_metric,
        'decoded_zero_phys': phys,
        'duration_s': duration,
        'final_losses': (L0m, L1m, L2m, L3m, L4m, L5m, Lzm)
    }


if __name__ == '__main__':
    overall_best = None
    # run for different scalers
    for scaler_name in ['robust', 'minmax']:
        scaler_results = []
        for lr in LRS:
            for k0 in K0S:
                for zw in ZWS:
                    for mode in MODES:
                        name = f"{scaler_name}_lr{lr}_k0{k0}_zw{zw}_mode{mode}"
                        print('\nStarting', name)
                        res = train_trial(lr, k0, zw, mode, name, scaler_name=scaler_name, epochs=150, num_sequences=32, seq_len=8)
                        scaler_results.append(res)
                        RESULTS.append(res)
                        print('Result:', res['trial'], 'mode', res['mode'], 'metric', res['best_metric'], 'decoded_zero', res['decoded_zero_phys'])
                        if overall_best is None or res['best_metric'] < overall_best['best_metric']:
                            overall_best = res
        # save per-scaler results
        import json
        with open(f'hp_tune_results_{scaler_name}.json', 'w') as f:
            json.dump(scaler_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        print(f'Saved per-scaler results to hp_tune_results_{scaler_name}.json')
    print('\nOverall best:', overall_best)
    # save results
    import json
    with open('hp_tune_results.json', 'w') as f:
        json.dump(RESULTS, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    print('Saved results to hp_tune_results.json')
