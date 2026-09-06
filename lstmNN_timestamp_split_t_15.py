import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- 1. Configurations and File Paths ---
BASE_DIR = r"c:\Users\vince\Desktop\NN"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets") if os.path.exists(os.path.join(BASE_DIR, "datasets")) else "datasets"

# The 6 dataset files with priority to W=60
DATASET_FILES = [
    {"name": "44misure_capped360_W60", "file": "NN_training_dataset_W60_44_capped360", "W": 60, "capped": True},
    {"name": "26misure_capped360_W60", "file": "NN_training_dataset_W60_26_capped360", "W": 60, "capped": True},
    {"name": "26misure_uncapped_W60",  "file": "NN_training_dataset_W60_26_uncapped",  "W": 60, "capped": False},
    {"name": "44misure_capped360_W30", "file": "NN_training_dataset_W30_44_capped360", "W": 30, "capped": True},
    {"name": "26misure_capped360_W30", "file": "NN_training_dataset_W30_26_capped360", "W": 30, "capped": True},
    {"name": "26misure_uncapped_W30",  "file": "NN_training_dataset_W30_26_uncapped",  "W": 30, "capped": False},
]

SPLIT_MODES = ["temporal", "patient_wise"]
EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 2e-2
EARLY_STOPPING_PATIENCE = 40
TTE_CAP_DAYS = 360.0


# --- 2. Feed-Forward Architecture (N -> H -> 1) ---
class SystematicFFNN(nn.Module):
    """
    Fixed N inputs (4 * M: slope a, intercept b, RMSE c, current value z),
    variable hidden nodes H (starting from 1, 2, 3 up to N),
    fixed 1 output predicting log(TTE).
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.20, noise_std=0.03):
        super().__init__()
        self.noise_std = noise_std
        self.hidden_dim = hidden_dim
        
        # When hidden_dim == 1, LayerNorm across dimension 1 would zero out outputs.
        # Thus, LayerNorm is applied only for hidden_dim > 1.
        if hidden_dim == 1:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(1, 1)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return self.net(x)


# --- 3. Data Loading & Feature Extraction (N = 4 * M) ---
def find_dataset_file(datasets_dir, base_name):
    candidates = [
        os.path.join(datasets_dir, base_name),
        os.path.join(datasets_dir, base_name + ".csv"),
        os.path.join(datasets_dir, base_name + ".CSV"),
        os.path.join(BASE_DIR, base_name),
        os.path.join(BASE_DIR, base_name + ".csv"),
        os.path.join(".", base_name),
        os.path.join(".", base_name + ".csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(datasets_dir, base_name + ".csv")


def load_dataset(file_path):
    print(f"\n[DATA LOADER] Reading file: {file_path}")
    t0 = time.time()
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns in {time.time() - t0:.2f}s.")
    
    # Dynamically resolve patient ID column spelling
    pid_col = 'partient_id' if 'partient_id' in df.columns else 'patient_id'
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if pid_col in df.columns:
            df = df.sort_values(by=[pid_col, 'timestamp']).reset_index(drop=True)
            
    return df, pid_col


def prepare_raw_data(df, pid_col):
    pid_all = df[pid_col].values
    time_all = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df))
    y_raw_days = df['tte'].values.astype(np.float32)
    
    # Target: use log_tte column directly if present, otherwise compute log(1 + TTE)
    if 'log_tte' in df.columns:
        y_log = df['log_tte'].values.astype(np.float32)
    else:
        y_log = np.log(np.maximum(y_raw_days, 1.0))
        
    all_str = ",".join(df['misure'].str.strip('{}'))
    parsed = np.fromstring(all_str, sep=',', dtype=np.float32)
    X_raw = parsed.reshape(len(df), -1)
    
    return X_raw, y_log, y_raw_days, pid_all, time_all


def create_split_masks(pid_all, time_all, split_mode="temporal", train_ratio=0.5, val_ratio=0.1):
    n_samples = len(pid_all)
    train_mask = np.zeros(n_samples, dtype=bool)
    val_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)
    unique_pids = np.unique(pid_all)
    
    if split_mode == "temporal":
        for pid in unique_pids:
            p_indices = np.where(pid_all == pid)[0]
            n_p = len(p_indices)
            n_tr = int(n_p * train_ratio)
            n_va = int(n_p * val_ratio)
            
            train_mask[p_indices[:n_tr]] = True
            val_mask[p_indices[n_tr : n_tr + n_va]] = True
            test_mask[p_indices[n_tr + n_va :]] = True
            
    elif split_mode == "patient_wise":
        n_pids = len(unique_pids)
        rng = np.random.RandomState(42)
        shuffled_pids = rng.permutation(unique_pids)
        
        n_tr_pids = int(n_pids * train_ratio)
        n_va_pids = int(n_pids * val_ratio)
        
        train_pids = set(shuffled_pids[:n_tr_pids])
        val_pids = set(shuffled_pids[n_tr_pids : n_tr_pids + n_va_pids])
        test_pids = set(shuffled_pids[n_tr_pids + n_va_pids :])
        
        for idx, pid in enumerate(pid_all):
            if pid in train_pids:
                train_mask[idx] = True
            elif pid in val_pids:
                val_mask[idx] = True
            elif pid in test_pids:
                test_mask[idx] = True
                
    return train_mask, val_mask, test_mask


def extract_features(X_all, y_log, y_days, pid_all, target_mask, W):
    """
    Checks if features are already extracted (104 or 176 features)
    or if sliding window of size W needs to be calculated over raw sessions (26 or 44 measures).
    Extracted features per measure: a (slope), b (intercept), c (RMSE), z (current value).
    """
    M_raw = X_all.shape[1]
    
    # Case A: Features are already pre-extracted in the CSV
    if M_raw in [104, 176]:
        valid_indices = np.where(target_mask)[0]
        return X_all[valid_indices], y_log[valid_indices].reshape(-1, 1), y_days[valid_indices].reshape(-1, 1)
        
    # Case B: Raw measures (26 or 44) -> compute sliding window W
    N_total = len(pid_all)
    pids_arr = np.array(pid_all)
    
    same_patient = (pids_arr[W - 1:] == pids_arr[: N_total - W + 1])
    target_in_set = target_mask[W - 1:]
    valid_flags = same_patient & target_in_set
    valid_indices = np.where(valid_flags)[0] + (W - 1)
    
    N_valid = len(valid_indices)
    if N_valid == 0:
        return np.empty((0, 4 * M_raw), dtype=np.float32), np.empty((0, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    t = np.arange(W, dtype=np.float32)
    t_mean = (W - 1) / 2.0
    t_dev = t - t_mean
    sum_t_dev_sq = np.sum(t_dev**2)
    
    windows = np.zeros((N_valid, W, M_raw), dtype=np.float32)
    for idx_out, target_idx in enumerate(valid_indices):
        windows[idx_out] = X_all[target_idx - W + 1 : target_idx + 1]
        
    z_mean = np.mean(windows, axis=1)
    a = np.sum(t_dev[:, None] * windows, axis=1) / sum_t_dev_sq
    b = z_mean - a * t_mean
    y_line = a[:, None, :] * t[None, :, None] + b[:, None, :]
    residuals = y_line - windows
    c_rmse = np.sqrt(np.mean(residuals**2, axis=1))
    z_curr = windows[:, -1, :]
    
    X_features = np.concatenate([a, b, c_rmse, z_curr], axis=1)
    return X_features, y_log[valid_indices].reshape(-1, 1), y_days[valid_indices].reshape(-1, 1)


# --- 4. Tabular Dataset & Evaluation ---
class TabularLogDataset(Dataset):
    def __init__(self, X_tab, y_log, y_days):
        self.X = torch.tensor(X_tab, dtype=torch.float32)
        self.y_log = torch.tensor(y_log, dtype=torch.float32)
        self.y_days = torch.tensor(y_days, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_log[idx], self.y_days[idx]


def evaluate_model(model, data_loader, device, is_capped):
    model.eval()
    total_mae_days = 0.0
    total_mape_days = 0.0
    total_mae_log = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for X_b, y_log_b, y_days_b in data_loader:
            X_b = X_b.to(device)
            y_log_b = y_log_b.to(device)
            y_days_b = y_days_b.to(device)
            
            preds_log = model(X_b)
            # Inversion: exp(pred_log) guarantees strictly positive TTE values
            preds_days = torch.exp(preds_log)
            
            # Apply 360-day cap ONLY if the dataset is capped
            if is_capped:
                preds_days = torch.clamp(preds_days, max=TTE_CAP_DAYS)
                
            mae_days = torch.abs(preds_days - y_days_b).sum().item()
            denom = torch.clamp(y_days_b, min=10.0)
            mape_days = (torch.abs(preds_days - y_days_b) / denom).sum().item() * 100.0
            mae_log = torch.abs(preds_log - y_log_b).sum().item()
            
            b_size = X_b.size(0)
            total_mae_days += mae_days
            total_mape_days += mape_days
            total_mae_log += mae_log
            total_samples += b_size
            
    if total_samples == 0:
        return 0.0, 0.0, 0.0
        
    return total_mae_days / total_samples, total_mape_days / total_samples, total_mae_log / total_samples


# --- 5. Training Loop with Epoch Early Stopping ---
def train_model(model, train_loader, val_loader, test_loader, device, is_capped, epochs=EPOCHS):
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_mae = float('inf')
    best_weights = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for X_b, y_log_b, _ in train_loader:
            X_b, y_log_b = X_b.to(device), y_log_b.to(device)
            optimizer.zero_grad()
            out_log = model(X_b)
            loss = criterion(out_log, y_log_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        scheduler.step()
        
        val_mae_d, _, _ = evaluate_model(model, val_loader, device, is_capped=is_capped)
        if val_mae_d < best_val_mae:
            best_val_mae = val_mae_d
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break
                
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        
    tr_mae, tr_mape, _ = evaluate_model(model, train_loader, device, is_capped=is_capped)
    va_mae, va_mape, _ = evaluate_model(model, val_loader, device, is_capped=is_capped)
    te_mae, te_mape, _ = evaluate_model(model, test_loader, device, is_capped=is_capped)
    
    return tr_mae, tr_mape, va_mae, va_mape, te_mae, te_mape


# --- 6. Architectural Pruning Controller ---
def check_architectural_pruning(history, max_patience=2, gap_threshold=35.0, gap_growth_ratio=1.35):
    if len(history) < 2:
        return False, ""
        
    curr = history[-1]
    prev = history[-2]
    gap_curr = abs(curr["val_mae"] - curr["train_mae"])
    gap_prev = abs(prev["val_mae"] - prev["train_mae"])
    
    # Check 1: Overfitting onset
    if gap_curr > gap_threshold and (gap_prev == 0 or gap_curr > gap_growth_ratio * gap_prev):
        return True, f"Overfitting trigger: |Val-Train| gap widened from {gap_prev:.1f}d to {gap_curr:.1f}d (> {gap_threshold}d)."
        
    # Check 2: Validation stagnation
    best_val = min(h["val_mae"] for h in history[:-1])
    if len(history) >= max_patience + 1:
        recent_vals = [h["val_mae"] for h in history[-max_patience:]]
        if all(v >= best_val + 2.0 for v in recent_vals):
            return True, f"Validation stagnation: no improvement over {best_val:.2f}d for {max_patience} consecutive configurations."
            
    return False, ""


# --- 7. Systematic Grid Search Engine ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=========================================================================")
    print(f" SYSTEMATIC FFNN LOG(TTE) EXPLORATION ENGINE - DEVICE: {device}")
    print(f"=========================================================================\n")
    
    all_results = []
    
    for ds_entry in DATASET_FILES:
        ds_name = ds_entry["name"]
        ds_file = ds_entry["file"]
        W = ds_entry["W"]
        is_capped = ds_entry["capped"]
        
        file_path = find_dataset_file(DATASETS_DIR, ds_file)
        if not os.path.exists(file_path):
            print(f"[SKIP] File not found: {file_path}. Proceeding to next.")
            continue
            
        cap_str = f"Capped at {TTE_CAP_DAYS}d" if is_capped else "Uncapped"
        print(f"\n{'='*90}\n DATASET: {ds_name} | Window: W={W} | Mode: {cap_str}\n{'='*90}")
        df, pid_col = load_dataset(file_path)
        X_raw, y_log, y_days, pid_all, time_all = prepare_raw_data(df, pid_col)
        
        for split_mode in SPLIT_MODES:
            print(f"\n>>> Split Mode: {split_mode.upper()} <<<")
            train_mask, val_mask, test_mask = create_split_masks(pid_all, time_all, split_mode=split_mode)
            
            # Extract 4 features per measure: a, b, c, z -> N = 4 * M
            X_tr, y_tr_log, y_tr_d = extract_features(X_raw, y_log, y_days, pid_all, train_mask, W)
            X_va, y_va_log, y_va_d = extract_features(X_raw, y_log, y_days, pid_all, val_mask, W)
            X_te, y_te_log, y_te_d = extract_features(X_raw, y_log, y_days, pid_all, test_mask, W)
            
            if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
                print(f"[SKIP] Insufficient samples for split {split_mode}.")
                continue
                
            mean = X_tr.mean(axis=0, keepdims=True)
            std = X_tr.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            
            X_tr_s = (X_tr - mean) / std
            X_va_s = (X_va - mean) / std
            X_te_s = (X_te - mean) / std
            
            train_loader = DataLoader(TabularLogDataset(X_tr_s, y_tr_log, y_tr_d), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(TabularLogDataset(X_va_s, y_va_log, y_va_d), batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(TabularLogDataset(X_te_s, y_te_log, y_te_d), batch_size=BATCH_SIZE, shuffle=False)
            
            N = X_tr_s.shape[1]
            M_meas = N // 4
            print(f"Verified Input Layer Dimension: N = 4 x {M_meas} = {N} features.")
            
            # Hidden units progression: explicitly includes 1, 2, 3 and scales up to N
            base_steps = [1, 2, 3, 5, 8, 16, 32, 64, 128]
            candidate_H = sorted(list(set([h for h in base_steps if h < N] + [N])))
            
            print(f"Hidden node search progression H: {candidate_H}")
            
            arch_history = []
            
            for h in candidate_H:
                model = SystematicFFNN(input_dim=N, hidden_dim=h).to(device)
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                tr_mae, tr_mape, va_mae, va_mape, te_mae, te_mape = train_model(
                    model, train_loader, val_loader, test_loader, device, is_capped=is_capped
                )
                
                diff_tr_te = abs(tr_mae - te_mae)
                
                res_entry = {
                    "Dataset": ds_name,
                    "Measures": M_meas,
                    "Capped": is_capped,
                    "Split": split_mode,
                    "Window_W": W,
                    "N_inputs": N,
                    "Hidden_H": h,
                    "Parameters": num_params,
                    "Train_MAE_days": round(tr_mae, 2),
                    "Val_MAE_days": round(va_mae, 2),
                    "Test_MAE_days": round(te_mae, 2),
                    "Diff_Train_Test": round(diff_tr_te, 2),
                    "Test_MAPE": round(te_mape, 2)
                }
                all_results.append(res_entry)
                arch_history.append({"H": h, "train_mae": tr_mae, "val_mae": va_mae, "test_mae": te_mae})
                
                print(f"  [H = {h:3d}] ({num_params:5d} params) -> Train MAE: {tr_mae:6.2f}d | Val MAE: {va_mae:6.2f}d | "
                      f"Test MAE: {te_mae:6.2f}d | |Train-Test| Gap: {diff_tr_te:5.2f}d | Test MAPE: {te_mape:5.1f}%")
                
                # Check for architectural pruning
                should_prune, reason = check_architectural_pruning(arch_history)
                if should_prune:
                    print(f"     ==> [PRUNING TRIGGERED] H expansion stopped for {ds_name} ({split_mode}). Reason: {reason}")
                    break

    # --- 8. Export and Display Comparative Summary ---
    df_results = pd.DataFrame(all_results)
    out_csv = os.path.join(BASE_DIR, "final_systematic_ffnn_log_tte_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\n{'='*100}\nSYSTEMATIC EXPLORATION COMPLETE! Results saved to:\n{out_csv}\n{'='*100}")
    
    if not df_results.empty:
        # Table 1: Best Ergodic Alignment
        print("\nTOP CONFIGURATIONS BY ERGODIC CONVERGENCE (|Train MAE - Test MAE|):")
        best_ergodic = df_results.sort_values(by="Diff_Train_Test").head(15)
        print(best_ergodic[["Dataset", "Measures", "Split", "Window_W", "Hidden_H", "Parameters", "Train_MAE_days", "Test_MAE_days", "Diff_Train_Test"]].to_string(index=False))
        
        # Table 2: Best Test Performance
        print("\nTOP CONFIGURATIONS BY LOWEST TEST MAE (Days):")
        best_test = df_results.sort_values(by="Test_MAE_days").head(15)
        print(best_test[["Dataset", "Measures", "Split", "Window_W", "Hidden_H", "Parameters", "Train_MAE_days", "Test_MAE_days", "Diff_Train_Test"]].to_string(index=False))


if __name__ == "__main__":
    main()