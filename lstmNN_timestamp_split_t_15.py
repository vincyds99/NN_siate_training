import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- Configurations ---
DEFAULT_CSV = r"c:\Users\vince\Desktop\NN\NN_training_dataset.csv"
CSV_PATH = DEFAULT_CSV if os.path.exists(DEFAULT_CSV) else r"c:\Users\vince\Desktop\NN\NN_training_dataset.csv"
CACHE_DIR = r"c:\Users\vince\Desktop\NN"

EPOCHS = 350
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2 # Heavy L2 regularization to prevent overfitting on tabular features

# Multi-Window configuration suggested by the professor (Step 2)
WINDOWS = [30, 60, 90]
SPLIT_MODE = "per_patient_temporal"

# --- 1. Custom Metrics & Loss ---
class MAPELoss(nn.Module):
    def __init__(self, min_val=10.0):
        super().__init__()
        self.min_val = min_val

    def forward(self, outputs, targets):
        denom = torch.clamp(targets, min=self.min_val)
        absolute_percentage_errors = torch.abs(outputs - targets) / denom
        return torch.mean(absolute_percentage_errors) * 100.0


# --- 2. Data Preparation ---
def load_and_preprocess_data():
    print(f"Parsing CSV file: {CSV_PATH}")
    t_start = time.time()
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded CSV file in {time.time() - t_start:.2f}s. Initial row count: {len(df)}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print("Sorting rows chronologically by patient_id and timestamp...")
    df = df.sort_values(by=['patient_id', 'timestamp']).reset_index(drop=True)
    
    def parse_vector_column(series):
        t0 = time.time()
        all_str = ",".join(series.str.strip('{}'))
        parsed = np.fromstring(all_str, sep=',', dtype=np.float32)
        reshaped = parsed.reshape(len(series), -1)
        print(f"Parsed column '{series.name}' in {time.time() - t0:.2f}s. Feature size: {reshaped.shape[1]}")
        return reshaped

    print("Parsing vector column 'misure'...")
    t_parse = time.time()
    misure = parse_vector_column(df['misure'])
    print(f"Total vector parsing completed in {time.time() - t_parse:.2f}s.")
    
    X_all = misure
    y_all = df['tte'].values.astype(np.float32)
    pid_all = df['patient_id'].values
    time_all = df['timestamp'].values
    
    return X_all, y_all, pid_all, time_all


# --- 3. Temporal Patient-wise Dataset Split Masks ---
def create_temporal_split_masks(pid_all, time_all, train_ratio=0.5, val_ratio=0.1, mode="per_patient_temporal"):
    n_samples = len(pid_all)
    train_mask = np.zeros(n_samples, dtype=bool)
    val_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)

    if mode == "per_patient_temporal":
        print("Executing Per-Patient Chronological Temporal Split (60% past / 40% future per patient)...")
        unique_pids = np.unique(pid_all)
        
        for pid in unique_pids:
            p_indices = np.where(pid_all == pid)[0]
            n_p = len(p_indices)
            
            n_train = int(n_p * train_ratio)
            n_val = int(n_p * val_ratio)
            
            train_idx_p = p_indices[:n_train]
            val_idx_p = p_indices[n_train : n_train + n_val]
            test_idx_p = p_indices[n_train + n_val :]
            
            train_mask[train_idx_p] = True
            val_mask[val_idx_p] = True
            test_mask[test_idx_p] = True

    print(f"Temporal Split Summary: Train = {train_mask.sum()} samples ({train_mask.mean()*100:.1f}%) | "
          f"Val = {val_mask.sum()} samples ({val_mask.mean()*100:.1f}%) | "
          f"Test = {test_mask.sum()} samples ({test_mask.mean()*100:.1f}%)")
          
    return train_mask, val_mask, test_mask


# --- 4. Multi-Window Linear Regression Feature Extractor (W=30, 60, 90) ---
def extract_multi_window_trend_features(X_all, y_all, pid_all, allowed_mask, windows=[30, 60, 90]):
    """
    Extracts multi-window linear regression features:
    For each W in windows: computes a_W, b_W, c_W (3 * M features).
    Appends current value z_curr (M features).
    Total features = (3 * len(windows) + 1) * M features per sample.
    """
    max_W = max(windows)
    print(f"Extracting Multi-Window Linear Regression Features {windows} (max W={max_W})...")
    t0 = time.time()
    
    pids_arr = np.array(pid_all)
    M = X_all.shape[1]
    
    # 1. Valid window indices: same patient & max_W steps strictly inside allowed_mask
    same_patient = (pids_arr[max_W - 1:] == pids_arr[: len(pids_arr) - max_W + 1])
    window_allowed = np.convolve(allowed_mask.astype(int), np.ones(max_W, dtype=int), mode='valid') == max_W
    valid_flags = same_patient & window_allowed
    valid_indices = np.where(valid_flags)[0] + (max_W - 1)
    
    N_valid = len(valid_indices)
    print(f"Extracted {N_valid} valid multi-window samples in {time.time() - t0:.2f}s.")
    
    if N_valid == 0:
        total_dim = (3 * len(windows) + 1) * M
        return np.empty((0, total_dim), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    features_list = []
    
    for W in windows:
        t = np.arange(W, dtype=np.float32)
        t_mean = (W - 1) / 2.0
        t_dev = t - t_mean
        sum_t_dev_sq = np.sum(t_dev**2)
        
        w_slices = np.zeros((N_valid, W, M), dtype=np.float32)
        for idx_out, target_idx in enumerate(valid_indices):
            w_slices[idx_out] = X_all[target_idx - W + 1 : target_idx + 1]
            
        z_mean = np.mean(w_slices, axis=1)
        a = np.sum(t_dev[:, None] * w_slices, axis=1) / sum_t_dev_sq
        b = z_mean - a * t_mean
        
        y_line = a[:, None, :] * t[None, :, None] + b[:, None, :]
        residuals = y_line - w_slices
        c = np.sqrt(np.mean(residuals**2, axis=1))
        
        features_list.extend([a, b, c])
        
    z_curr = X_all[valid_indices]
    features_list.append(z_curr)
    
    X_features = np.concatenate(features_list, axis=1)
    y_targets = y_all[valid_indices].reshape(-1, 1)
    
    return X_features, y_targets


# --- 5. PyTorch Tabular Dataset ---
class TrendTabularDataset(Dataset):
    def __init__(self, X_tab, y_tab):
        self.X = torch.tensor(X_tab, dtype=torch.float32)
        self.y = torch.tensor(y_tab, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- 6. MULTI-WINDOW FEED-FORWARD NEURAL NETWORK ARCHITECTURE ---
class MultiWindowTrendFFNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.35):
        """
        Deep Regularized FFNN designed to process multi-window trend features
        (a_30, b_30, c_30, a_60, b_60, c_60, a_90, b_90, c_90, z_curr).
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --- Early Stopping Helper ---
class EarlyStoppingMAE:
    def __init__(self, patience=30, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_mae = None
        self.early_stop = False

    def __call__(self, val_mae):
        if self.best_mae is None:
            self.best_mae = val_mae
        elif val_mae > self.best_mae - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_mae = val_mae
            self.counter = 0


# --- Evaluation Function ---
def evaluate_dataset(model, data_loader, device):
    model.eval()
    mape_criterion = MAPELoss(min_val=10.0)
    total_mape = 0.0
    total_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            
            mape_val = mape_criterion(outputs, y_batch)
            mae_val = torch.abs(outputs - y_batch).sum()
            
            batch_size = X_batch.size(0)
            total_mape += mape_val.item() * batch_size
            total_mae += mae_val.item()
            total_samples += batch_size
            
    avg_mape = total_mape / total_samples
    avg_mae = total_mae / total_samples
    return avg_mape, avg_mae


# --- 7. Training Loop ---
def train_and_evaluate_ffnn(model, train_loader, val_loader, test_loader, exp_config, exp_name, device):
    train_criterion = nn.SmoothL1Loss(beta=2.0) # Huber Loss for stable trend regression
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    val_mae_hist = []
    best_val_mae = float('inf')
    early_stopping = EarlyStoppingMAE(patience=30)
    weights_path = os.path.join(CACHE_DIR, f"best_weights_{exp_name}.pth")
    
    print(f"\n--- Training {exp_name} on {device} ---")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            loss = train_criterion(outputs, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item() * X_batch.size(0)
            
        scheduler.step()
        epoch_train_loss /= len(train_loader.dataset)
        
        val_mape, val_mae = evaluate_dataset(model, val_loader, device)
        val_mae_hist.append(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), weights_path)
            
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 5 == 0 or epoch == 0:
            train_mape, train_mae = evaluate_dataset(model, train_loader, device)
            print(f"Epoch {epoch+1:03d}/{EPOCHS:03d} | LR: {current_lr:.1e} | "
                  f"Train MAE: {train_mae:.2f}d | Val MAE: {val_mae:.2f}d (Best Val: {best_val_mae:.2f}d)")
        
        early_stopping(val_mae)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
            
    print(f"\n[EVALUATION] Loading optimal weights from '{weights_path}'...")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        
    final_train_mape, final_train_mae = evaluate_dataset(model, train_loader, device)
    final_val_mape, final_val_mae = evaluate_dataset(model, val_loader, device)
    final_test_mape, final_test_mae = evaluate_dataset(model, test_loader, device)
    
    print(f"--> [MULTI-WINDOW RESULT] {exp_name} | "
          f"Train MAE: {final_train_mae:.2f}d | Val MAE: {final_val_mae:.2f}d | Test MAE: {final_test_mae:.2f}d | "
          f"Test MAPE: {final_test_mape:.2f}%")
    
    return val_mae_hist, final_train_mape, final_train_mae, final_val_mape, final_val_mae, final_test_mape, final_test_mae


# --- 8. Main Execution Pipeline ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution device: {device}")
    
    # 1. Load Data
    X_all, y_all, pid_all, time_all = load_and_preprocess_data()
    
    # 2. Temporal Patient-wise Split
    train_mask, val_mask, test_mask = create_temporal_split_masks(
        pid_all, time_all, train_ratio=0.5, val_ratio=0.1, mode=SPLIT_MODE
    )
    
    # 3. Extract Multi-Window Linear Regression Features (W = [30, 60, 90])
    X_train_raw, y_train = extract_multi_window_trend_features(X_all, y_all, pid_all, train_mask, windows=WINDOWS)
    X_val_raw, y_val = extract_multi_window_trend_features(X_all, y_all, pid_all, val_mask, windows=WINDOWS)
    X_test_raw, y_test = extract_multi_window_trend_features(X_all, y_all, pid_all, test_mask, windows=WINDOWS)
    
    # Standardization strictly computed on Train partition
    mean = X_train_raw.mean(axis=0, keepdims=True)
    std = X_train_raw.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    
    X_train_scaled = (X_train_raw - mean) / std
    X_val_scaled = (X_val_raw - mean) / std
    X_test_scaled = (X_test_raw - mean) / std
    
    # 4. Create PyTorch DataLoaders
    train_dataset = TrendTabularDataset(X_train_scaled, y_train)
    val_dataset = TrendTabularDataset(X_val_scaled, y_val)
    test_dataset = TrendTabularDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = X_train_scaled.shape[1]
    print(f"Multi-Window Tabular Input Dimension (W={WINDOWS}): {input_dim} features.")
    
    # 5. Define Multi-Window FFNN Experiments
    experiments = [
        {
            "name": f"MultiWindowFFNN_Standard_W30_60_90",
            "hidden_dims": [128, 64, 32],
            "dropout": 0.30
        },
        {
            "name": f"MultiWindowFFNN_Compact_W30_60_90",
            "hidden_dims": [64, 32],
            "dropout": 0.25
        },
        {
            "name": f"MultiWindowFFNN_Deep_W30_60_90",
            "hidden_dims": [256, 128, 64, 32],
            "dropout": 0.35
        }
    ]
    
    results = []
    val_histories = {}
    
    for exp in experiments:
        name = exp["name"]
        print(f"\nConfiguring Experiment: {name}...")
        
        model = MultiWindowTrendFFNN(
            input_dim=input_dim,
            hidden_dims=exp["hidden_dims"],
            dropout_rate=exp["dropout"]
        ).to(device)
        
        val_mae_hist, train_mape, train_mae, val_mape, val_mae, test_mape, test_mae = train_and_evaluate_ffnn(
            model, train_loader, val_loader, test_loader, exp, name, device
        )
        
        val_histories[name] = val_mae_hist
        results.append({
            "Architecture / Pipeline": name,
            "Train MAE (days)": round(train_mae, 2),
            "Val MAE (days)": round(val_mae, 2),
            "Test MAE (days)": round(test_mae, 2),
            "Train MAPE (%)": round(train_mape, 2),
            "Val MAPE (%)": round(val_mape, 2),
            "Test MAPE (%)": round(test_mape, 2)
        })
        
    # --- GENERATE FINAL TABLE ---
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*110)
    print(f" FINAL PERFORMANCE EVALUATION TABLE (MULTI-WINDOW FFNN MODEL, WINDOWS = {WINDOWS})")
    print("="*110)
    print(df_results.to_string(index=False))
    print("="*110)
    
    # Save table to CSV
    csv_out_path = os.path.join(CACHE_DIR, f"final_test_performance_multi_window_W30_60_90.csv")
    df_results.to_csv(csv_out_path, index=False)
    print(f"\nResults table saved to CSV: {csv_out_path}")
    
    # --- PLOT 1: FINAL TEST MAE BAR CHART ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_results["Architecture / Pipeline"], df_results["Test MAE (days)"], color=['#2b5c8f', '#d95f02', '#7570b3'])
    plt.title(f"Final Test MAE Evaluation in Days (Multi-Window FFNN, W = {WINDOWS})", fontsize=13, fontweight='bold')
    plt.ylabel("Mean Absolute Error (MAE) in Days", fontsize=11)
    plt.xticks(rotation=10, ha="right")
    
    max_mae = df_results["Test MAE (days)"].max()
    plt.ylim(0, max_mae * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max_mae * 0.02), f"{yval:.2f}d", ha='center', va='bottom', fontweight='bold')
        
    plt.subplots_adjust(bottom=0.25, top=0.90)
    bar_plot_path = os.path.join(CACHE_DIR, f"final_test_mae_multi_window_W30_60_90.png")
    plt.savefig(bar_plot_path, bbox_inches='tight')
    print(f"Performance bar chart saved to: {bar_plot_path}")
    
    # --- PLOT 2: VALIDATION LEARNING CURVES ---
    plt.figure(figsize=(10, 6))
    for name, hist in val_histories.items():
        plt.plot(range(1, len(hist) + 1), hist, label=f"{name}")
        
    plt.title(f"Validation MAE Curves per Epoch in Days (Multi-Window FFNN, W = {WINDOWS})", fontsize=13)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation MAE (Days)", fontsize=11)
    plt.legend()
    plt.grid(True, ls="--")
    plt.tight_layout()
    curve_plot_path = os.path.join(CACHE_DIR, f"val_mae_learning_curves_multi_window_W30_60_90.png")
    plt.savefig(curve_plot_path)
    print(f"Validation learning curves plot saved to: {curve_plot_path}")

if __name__ == "__main__":
    main()