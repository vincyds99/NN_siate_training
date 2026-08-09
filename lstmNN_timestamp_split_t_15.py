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

EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 5e-5

# Split Configuration: "per_patient_temporal" (60% past / 40% future per patient)
SPLIT_MODE = "per_patient_temporal" 

# --- 1. Custom Metrics & Loss Functions ---
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


# --- 3. Temporal Patient-wise Dataset Split Function ---
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


# --- 4. Custom Dataset with Feature Delta Augmentation (Delta X) ---
class EnhancedDialysisDataset(Dataset):
    def __init__(self, X, y, pids, allowed_mask, T, use_deltas=True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.T = T
        self.use_deltas = use_deltas
        
        pids_arr = np.array(pids)
        same_patient = pids_arr[T - 1:] == pids_arr[:- (T - 1)]
        target_mask = allowed_mask[T - 1:]
        
        valid_flags = same_patient & target_mask
        self.valid_indices = np.where(valid_flags)[0] + (T - 1)
        
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        target_idx = self.valid_indices[idx]
        X_seq = self.X[target_idx - self.T + 1 : target_idx + 1]
        
        if self.use_deltas:
            deltas = torch.zeros_like(X_seq)
            deltas[1:] = X_seq[1:] - X_seq[:-1]
            X_seq = torch.cat([X_seq, deltas], dim=-1)
            
        y_target = self.y[target_idx]
        return X_seq, y_target


# --- 5. COMPACT ARCHITECTURE: CompactBiLSTM ---
class CompactBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size=16, dropout_rate=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        concat_dim = hidden_size * 2
        self.head = nn.Sequential(
            nn.Linear(concat_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.head(last_step)


# --- Early Stopping Helper (Monitoring MAE in Days) ---
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


# --- 6. Training Loop (Optimizing Pure MAE Loss) ---
def train_and_evaluate_bilstm_mae(model, train_loader, val_loader, test_loader, exp_config, exp_name, device):
    # OPTION A: Pure MAE Loss (L1 Loss in days)
    train_criterion = nn.L1Loss()
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    if exp_config["scheduler"] == "warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    val_mae_hist = []
    best_val_mae = float('inf')
    early_stopping = EarlyStoppingMAE(patience=30)
    weights_path = f"best_weights_mae_{exp_name}.pth"
    
    print(f"\n--- Training Compact {exp_name} with Pure MAE Loss on {device} ---")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            loss = train_criterion(outputs, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_train_loss += loss.item() * X_batch.size(0)
            
        scheduler.step()
        epoch_train_loss /= len(train_loader.dataset)
        
        # Validation evaluation at epoch end
        val_mape, val_mae = evaluate_dataset(model, val_loader, device)
        val_mae_hist.append(val_mae)
        
        # Checkpointing based on Validation MAE (in days)
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
            
    # --- FORMAL EVALUATION ON TRAIN, VAL AND TEST SET ---
    print(f"\n[EVALUATION] Loading optimal weights from '{weights_path}' for evaluation...")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        
    final_train_mape, final_train_mae = evaluate_dataset(model, train_loader, device)
    final_val_mape, final_val_mae = evaluate_dataset(model, val_loader, device)
    final_test_mape, final_test_mae = evaluate_dataset(model, test_loader, device)
    
    print(f"--> [FINAL OPTION A RESULT] {exp_name} | "
          f"Train MAE: {final_train_mae:.2f}d | Val MAE: {final_val_mae:.2f}d | Test MAE: {final_test_mae:.2f}d | "
          f"Test MAPE: {final_test_mape:.2f}%")
    
    return val_mae_hist, final_train_mape, final_train_mae, final_val_mape, final_val_mae, final_test_mape, final_test_mae


# --- 7. Main Execution Pipeline ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")
    
    # 1. Load Data
    X_all, y_all, pid_all, time_all = load_and_preprocess_data()
    
    # 2. Temporal Patient-wise Split (60% Train+Val / 40% Test)
    train_mask, val_mask, test_mask = create_temporal_split_masks(
        pid_all, time_all, train_ratio=0.5, val_ratio=0.1, mode=SPLIT_MODE
    )
    
    # Standardization computed strictly on the Training partition
    mean = X_all[train_mask].mean(axis=0, keepdims=True)
    std = X_all[train_mask].std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_scaled = (X_all - mean) / std
    
    seq_len = 15
    print(f"Sliding window length configured to T = {seq_len} sessions.")
    
    # 3. Define Experiments (Evaluating Pure MAE Training)
    experiments = [
        {
            "name": "BiLSTM_16_1L_MAE_Baseline",
            "use_deltas": False,
            "scheduler": "cosine"
        },
        {
            "name": "BiLSTM_16_1L_MAE_FeatureDeltas",
            "use_deltas": True,
            "scheduler": "cosine"
        },
        {
            "name": "BiLSTM_16_1L_MAE_WarmRestarts",
            "use_deltas": False,
            "scheduler": "warm_restarts"
        },
        {
            "name": "BiLSTM_16_1L_MAE_Deltas_WarmRestarts",
            "use_deltas": True,
            "scheduler": "warm_restarts"
        }
    ]
    
    results = []
    val_histories = {}
    
    for exp in experiments:
        name = exp["name"]
        print(f"\nConfiguring Experiment: {name}...")
        
        train_dataset = EnhancedDialysisDataset(X_scaled, y_all, pid_all, train_mask, T=seq_len, use_deltas=exp["use_deltas"])
        val_dataset = EnhancedDialysisDataset(X_scaled, y_all, pid_all, val_mask, T=seq_len, use_deltas=exp["use_deltas"])
        test_dataset = EnhancedDialysisDataset(X_scaled, y_all, pid_all, test_mask, T=seq_len, use_deltas=exp["use_deltas"])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        num_features = X_scaled.shape[1]
        input_dim = num_features * 2 if exp["use_deltas"] else num_features
        
        # Instantiate Reduced Compact Architecture
        model = CompactBiLSTM(input_dim=input_dim, hidden_size=16, dropout_rate=0.2).to(device)
        
        val_mae_hist, train_mape, train_mae, val_mape, val_mae, test_mape, test_mae = train_and_evaluate_bilstm_mae(
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
    print(f" FINAL TEST SET PERFORMANCE EVALUATION TABLE (OPTION A - PURE MAE LOSS, T = {seq_len})")
    print("="*110)
    print(df_results.to_string(index=False))
    print("="*110)
    
    # Save table to CSV
    csv_out_path = os.path.join(CACHE_DIR, f"final_test_performance_optionA_MAE_T{seq_len}.csv")
    df_results.to_csv(csv_out_path, index=False)
    print(f"\nResults table saved to CSV: {csv_out_path}")
    
    # --- PLOT 1: FINAL TEST MAE BAR CHART (IN DAYS) ---
    plt.figure(figsize=(12, 7))
    bars = plt.bar(df_results["Architecture / Pipeline"], df_results["Test MAE (days)"], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title(f"Final Test MAE Evaluation in Days (Option A - Pure MAE Loss, T = {seq_len})", fontsize=14, fontweight='bold')
    plt.ylabel("Mean Absolute Error (MAE) in Days", fontsize=12)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    
    max_mae = df_results["Test MAE (days)"].max()
    plt.ylim(0, max_mae * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max_mae * 0.02), f"{yval:.2f}d", ha='center', va='bottom', fontweight='bold')
        
    plt.subplots_adjust(bottom=0.25, top=0.90)
    bar_plot_path = os.path.join(CACHE_DIR, f"final_test_mae_optionA_T{seq_len}.png")
    plt.savefig(bar_plot_path, bbox_inches='tight')
    print(f"Performance bar chart saved to: {bar_plot_path}")
    
    # --- PLOT 2: VALIDATION LEARNING CURVES (MAE IN DAYS) ---
    plt.figure(figsize=(12, 7))
    for name, hist in val_histories.items():
        plt.plot(range(1, len(hist) + 1), hist, label=f"{name}")
        
    plt.title(f"Validation MAE Curves per Epoch in Days (Option A, T = {seq_len})", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation MAE (Days)", fontsize=12)
    plt.legend()
    plt.grid(True, ls="--")
    plt.tight_layout()
    curve_plot_path = os.path.join(CACHE_DIR, f"val_mae_learning_curves_optionA_T{seq_len}.png")
    plt.savefig(curve_plot_path)
    print(f"Validation learning curves plot saved to: {curve_plot_path}")

if __name__ == "__main__":
    main()