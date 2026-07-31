import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- Configurations ---
CSV_PATH = r"c:\Users\vince\Desktop\NN\NN_training_dataset.csv"
CACHE_DIR = r"c:\Users\vince\Desktop\NN"

# Cache paths for the preprocessed LSTM pipeline
X_ALL_PATH = os.path.join(CACHE_DIR, "X_all_lstm_misure.npy")
Y_ALL_PATH = os.path.join(CACHE_DIR, "y_all_lstm_misure.npy")
PID_ALL_PATH = os.path.join(CACHE_DIR, "pid_all_lstm_misure.npy")
TIME_ALL_PATH = os.path.join(CACHE_DIR, "time_all_lstm_misure.npy")

EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 5e-5

# --- 1. Custom Loss Functions ---
class MAPELoss(nn.Module):
    def __init__(self, min_val=10.0):
        super().__init__()
        self.min_val = min_val

    def forward(self, outputs, targets):
        denom = torch.clamp(targets, min=self.min_val)
        absolute_percentage_errors = torch.abs(outputs - targets) / denom
        return torch.mean(absolute_percentage_errors) * 100.0


class HybridMAPEMAELoss(nn.Module):
    def __init__(self, min_val=10.0, mae_weight=0.01):
        """
        Hybrid Loss: Calculates MAPE with denominator clamping and adds a
        weighted penalty on mean absolute error (MAE) in days.
        """
        super().__init__()
        self.min_val = min_val
        self.mae_weight = mae_weight

    def forward(self, outputs, targets):
        denom = torch.clamp(targets, min=self.min_val)
        mape = torch.mean(torch.abs(outputs - targets) / denom) * 100.0
        mae = torch.mean(torch.abs(outputs - targets))
        return mape + (self.mae_weight * mae)


# --- 2. Data Preparation and Caching ---
def load_and_preprocess_data():
    cache_exists = all(os.path.exists(p) for p in [X_ALL_PATH, Y_ALL_PATH, PID_ALL_PATH, TIME_ALL_PATH])
    
    if cache_exists:
        print("Loading preprocessed dataset from LSTM numpy cache files...")
        t0 = time.time()
        X_all = np.load(X_ALL_PATH, allow_pickle=True)
        y_all = np.load(Y_ALL_PATH)
        pid_all = np.load(PID_ALL_PATH, allow_pickle=True)
        time_all = np.load(TIME_ALL_PATH, allow_pickle=True)
        print(f"Loaded datasets from cache in {time.time() - t0:.2f} seconds.")
        return X_all, y_all, pid_all, time_all

    print(f"Cache files not found. Parsing CSV file: {CSV_PATH}")
    t_start = time.time()
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded CSV file in {time.time() - t_start:.2f}s. Initial row count: {len(df)}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print("Sorting rows by patient_id and chronologically by timestamp...")
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
    
    print("Caching global preprocessed arrays for LSTM...")
    np.save(X_ALL_PATH, X_all)
    np.save(Y_ALL_PATH, y_all)
    np.save(PID_ALL_PATH, pid_all)
    np.save(TIME_ALL_PATH, time_all)
    print("Preprocessed dataset successfully saved to cache.")
    
    return X_all, y_all, pid_all, time_all


# --- 3. Custom Dataset with Feature Delta Augmentation (Delta X) ---
class EnhancedDialysisDataset(Dataset):
    def __init__(self, X, y, pids, allowed_pids, T, use_deltas=True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.T = T
        self.use_deltas = use_deltas
        
        mask = np.isin(pids, allowed_pids)
        pids_arr = np.array(pids)
        same_patient = pids_arr[T - 1:] == pids_arr[:- (T - 1)]
        target_mask = mask[T - 1:]
        
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


# --- 4. FIXED ARCHITECTURE: BiLSTM_32_1L ---
class FixedBiLSTM(nn.Module):
    def __init__(self, input_dim):
        """
        BiLSTM Architecture with 32 units per direction (64 total) with LayerNorm.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.head(last_step)


# --- Early Stopping Helper ---
class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# --- Evaluation Function ---
def evaluate_dataset(model, data_loader, device):
    """
    Performs formal evaluation on a DataLoader (Validation or Test Set).
    Returns MAPE (%) and MAE (days).
    """
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


# --- 5. Training Loop ---
def train_and_evaluate_bilstm(model, train_loader, val_loader, test_loader, exp_config, exp_name, device):
    if exp_config["loss_type"] == "hybrid":
        train_criterion = HybridMAPEMAELoss(min_val=10.0, mae_weight=0.01)
    else:
        train_criterion = HybridMAPEMAELoss(min_val=10.0, mae_weight=0.0)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    if exp_config["scheduler"] == "warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    train_losses, val_mape_hist = [], []
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=30)
    weights_path = f"best_weights_{exp_name}.pth"
    
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
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_train_loss += loss.item() * X_batch.size(0)
            
        scheduler.step()
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation at epoch end
        val_mape, val_mae = evaluate_dataset(model, val_loader, device)
        val_mape_hist.append(val_mape)
        
        # Checkpointing based on Validation Loss
        if val_mape < best_val_loss:
            best_val_loss = val_mape
            torch.save(model.state_dict(), weights_path)
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d}/{EPOCHS:03d} | LR: {current_lr:.1e} | Train Loss: {epoch_train_loss:.2f} | "
              f"Val MAPE: {val_mape:.2f}% (Best Val: {best_val_loss:.2f}%)")
        
        early_stopping(val_mape)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
            
    # --- FORMAL EVALUATION ON FINAL TEST SET ---
    print(f"\n[EVALUATION] Loading optimal weights from '{weights_path}' for evaluation on Test Set...")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        
    final_test_mape, final_test_mae = evaluate_dataset(model, test_loader, device)
    final_val_mape, final_val_mae = evaluate_dataset(model, val_loader, device)
    
    print(f"--> [FORMAL TEST SET RESULT] {exp_name} | MAPE: {final_test_mape:.2f}% | MAE: {final_test_mae:.2f} days")
    
    return val_mape_hist, final_val_mape, final_val_mae, final_test_mape, final_test_mae


# --- 6. Main Execution Pipeline ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")
    
    # 1. Load Data
    X_all, y_all, pid_all, time_all = load_and_preprocess_data()
    
    # 2. Patient-wise Split (50% train / 10% val / 40% test)
    unique_pids = np.unique(pid_all)
    np.random.seed(42)
    shuffled_pids = unique_pids.copy()
    np.random.shuffle(shuffled_pids)
    
    n_patients = len(shuffled_pids)
    train_end = int(n_patients * 0.5)
    val_end = int(n_patients * 0.6)
    
    train_pids = shuffled_pids[:train_end]
    val_pids = shuffled_pids[train_end:val_end]
    test_pids = shuffled_pids[val_end:]
    
    print(f"\nPatient-wise split summary: Total Unique Patients = {n_patients}")
    print(f"Train: {len(train_pids)} patients | Val: {len(val_pids)} patients | Test: {len(test_pids)} patients")
    
    train_mask = np.isin(pid_all, train_pids)
    
    # Standardization
    mean = X_all[train_mask].mean(axis=0, keepdims=True)
    std = X_all[train_mask].std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X_scaled = (X_all - mean) / std
    
    # --- SLIDING WINDOW SET TO 25 SESSIONS ---
    seq_len = 25
    print(f"Sliding window length configured to T = {seq_len} sessions.")
    
    # 3. Define the 4 Pipeline Experiments
    experiments = [
        {
            "name": "BiLSTM_32_1L_Winner_Baseline",
            "use_deltas": False,
            "loss_type": "pure_mape",
            "scheduler": "cosine"
        },
        {
            "name": "BiLSTM_32_1L_FeatureDeltas",
            "use_deltas": True,
            "loss_type": "pure_mape",
            "scheduler": "cosine"
        },
        {
            "name": "BiLSTM_32_1L_WarmRestarts",
            "use_deltas": False,
            "loss_type": "pure_mape",
            "scheduler": "warm_restarts"
        },
        {
            "name": "BiLSTM_32_1L_Deltas_HybridLoss",
            "use_deltas": True,
            "loss_type": "hybrid",
            "scheduler": "warm_restarts"
        }
    ]
    
    results = []
    val_histories = {}
    
    for exp in experiments:
        name = exp["name"]
        print(f"\nConfiguring Experiment: {name}...")
        
        train_dataset = EnhancedDialysisDataset(X_scaled, y_all, pid_all, train_pids, T=seq_len, use_deltas=exp["use_deltas"])
        val_dataset = EnhancedDialysisDataset(X_scaled, y_all, pid_all, val_pids, T=seq_len, use_deltas=exp["use_deltas"])
        test_dataset = EnhancedDialysisDataset(X_scaled, y_all, pid_all, test_pids, T=seq_len, use_deltas=exp["use_deltas"])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        num_features = X_scaled.shape[1]
        input_dim = num_features * 2 if exp["use_deltas"] else num_features
        model = FixedBiLSTM(input_dim=input_dim).to(device)
        
        val_mape_hist, val_mape, val_mae, test_mape, test_mae = train_and_evaluate_bilstm(
            model, train_loader, val_loader, test_loader, exp, name, device
        )
        
        val_histories[name] = val_mape_hist
        results.append({
            "Architecture / Pipeline": name,
            "Val MAPE (%)": round(val_mape, 2),
            "Val MAE (days)": round(val_mae, 2),
            "Test MAPE (%)": round(test_mape, 2),
            "Test MAE (days)": round(test_mae, 2)
        })
        
    # --- 1. GENERATE FINAL TABLE ---
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*95)
    print(f" FINAL TEST SET PERFORMANCE EVALUATION TABLE (T = {seq_len} SESSIONS)")
    print("="*95)
    print(df_results.to_string(index=False))
    print("="*95)
    
    # Save table to CSV
    csv_out_path = os.path.join(CACHE_DIR, f"final_test_performance_T{seq_len}.csv")
    df_results.to_csv(csv_out_path, index=False)
    print(f"\nResults table saved to CSV: {csv_out_path}")
    
    # --- 2. PLOT 1: FINAL TEST MAPE BAR CHART ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_results["Architecture / Pipeline"], df_results["Test MAPE (%)"], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title(f"Final Performance Evaluation on Test Set (T = {seq_len} Sessions)", fontsize=14, fontweight='bold')
    plt.ylabel("Mean Absolute Percentage Error (MAPE) in %", fontsize=12)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.ylim(50, 90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add numeric labels above each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    bar_plot_path = os.path.join(CACHE_DIR, f"final_test_mape_comparison_T{seq_len}.png")
    plt.savefig(bar_plot_path)
    print(f"Performance bar chart saved to: {bar_plot_path}")
    
    # --- 3. PLOT 2: VALIDATION LEARNING CURVES ---
    plt.figure(figsize=(12, 7))
    for name, hist in val_histories.items():
        plt.plot(range(1, len(hist) + 1), hist, label=f"{name}")
        
    plt.title(f"Validation Curves per Epoch (Validation MAPE %) - T = {seq_len} Sessions", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation MAPE (%)", fontsize=12)
    plt.legend()
    plt.grid(True, ls="--")
    plt.tight_layout()
    curve_plot_path = os.path.join(CACHE_DIR, f"val_mape_learning_curves_T{seq_len}.png")
    plt.savefig(curve_plot_path)
    print(f"Validation learning curves plot saved to: {curve_plot_path}")

if __name__ == "__main__":
    main()